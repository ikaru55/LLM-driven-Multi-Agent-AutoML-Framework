import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple

class CurricularFaceLoss(nn.Module):
    """
    CurricularFace-style Loss Implementation for Face Verification.

    Adheres to Hacker's Strategy:
    - Base Margin (Arc): m=0.5
    - Scale (s): 64.0
    - Numerical Stability: Strict clamping of cosine values to prevent NaN.
    - Architecture: Designed to work with the MetricHead's scaled output.

    This implementation focuses on the Arc-margin component which is the 
    core driver of class separation, ensuring stability over complex dynamic 
    curriculum terms that often cause gradient explosion in early training.
    """
    def __init__(self, s=64.0, m=0.5):
        super(CurricularFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.epsilon = 1e-6

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Scaled cosine similarity from MetricHead (B, C). 
                    Expected input is already s * cos(theta).
            labels: Ground truth class indices (B).
        """
        # 1. Recover raw cosine from scaled logits
        cosine = logits / self.s

        # 2. Strict Clamping for Numerical Stability (Hacker's requirement)
        cosine = torch.clamp(cosine, -1.0 + self.epsilon, 1.0 - self.epsilon)

        # 3. Calculate Angular Margin Term
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. Handle "Hard" Angles (Stability fallback)
        # If theta > pi - m, the margin function can become non-monotonic or unstable.
        # We fallback to a Taylor expansion approximation (cosine - mm) in that region.
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # 5. Create One-Hot Mask
        # We only apply the margin penalty to the ground truth class.
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 6. Combine
        # Target Class: phi (cos(theta+m))
        # Non-Target: cosine (cos(theta))
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 7. Rescale
        output *= self.s

        # 8. Cross Entropy
        return F.cross_entropy(output, labels)

class TrainerAdapter:
    """
    Trainer Adapter implementing the "Hacker's Refined Strategy" (Cycle 18).

    Core Optimization Strategy:
    - Optimizer: SGD + Momentum + Nesterov (The "Gold Standard" for ArcFace).
    - Scheduler: Linear Warmup (4 Epochs) -> Cosine Decay.
                 This addresses the "Optimization Instability" from Cycle 16.
    - Loss: CurricularFace (Arc Margin) with s=64, m=0.5.
    - Regularization: In-Trainer Mixup (Beta=1.0) to smooth decision boundaries.
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # --- Strategy Configuration ---
        self.base_lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.warmup_epochs = 4

        # --- State ---
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        train_cfg = self.config.get('train', {})
        self.epochs = int(train_cfg.get('epochs_full', 10))

    def setup_optimization(self, steps_per_epoch):
        """
        Configures the optimization engine with parameter groups and custom scheduling.
        """
        # 1. Parameter Groups: Separate Bias/BN from Weight Decay
        # This is crucial for training GhostNet/MobileFaceNet architectures effectively.
        param_groups = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bn' in name or 'bias' in name:
                # No weight decay for normalization and bias layers
                param_groups.append({'params': param, 'weight_decay': 0.0})
            else:
                param_groups.append({'params': param, 'weight_decay': self.weight_decay})

        # 2. Optimizer: SGD with Nesterov
        # Chosen over AdamW for better convergence in margin-based metric learning.
        self.optimizer = optim.SGD(param_groups, lr=self.base_lr, 
                                   momentum=self.momentum, nesterov=True)

        # 3. Scheduler: Linear Warmup -> Cosine Annealing
        total_steps = self.epochs * steps_per_epoch
        # Ensure warmup doesn't exceed total steps (e.g., in short test runs)
        warmup_steps = min(self.warmup_epochs * steps_per_epoch, int(0.5 * total_steps))

        def lr_lambda(current_step):
            # Phase 1: Linear Warmup (0 -> base_lr)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Phase 2: Cosine Decay (base_lr -> 0)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # 4. Loss Function
        self.criterion = CurricularFaceLoss(s=64.0, m=0.5).to(self.device)

        print(f'[Trainer] Strategy Applied (Cycle 18 Refined):')
        print(f'  • Optimizer : SGD (lr={self.base_lr}, mom={self.momentum}, wd={self.weight_decay})')
        print(f'  • Scheduler : Linear Warmup ({warmup_steps} steps) -> Cosine Decay')
        print(f'  • Loss      : CurricularFace (s=64.0, m=0.5)')
        print(f'  • Reg       : Mixup (alpha=1.0, In-Batch)')

    def run_epoch(self, loader, epoch, scaler, log_formatter) -> Tuple[float, float, float]:
        """
        Executes one training epoch with Mixup and Mixed Precision.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        max_grad_norm = 0.0

        # Handle Tqdm for progress visualization
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc=f'Ep {epoch}', leave=False)
        except ImportError:
            iterator = loader

        for imgs, labels in iterator:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # --- In-Trainer Mixup Logic ---
            # As per strategy, we apply Mixup here to smooth the decision boundary.
            # We sample from Beta(1.0, 1.0) which is a uniform distribution.
            use_mixup = True
            alpha = 1.0

            if use_mixup:
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(imgs.size(0)).to(self.device)

                # Mix images
                mixed_imgs = lam * imgs + (1 - lam) * imgs[index]

                # Keep both labels for loss calculation
                label_a, label_b = labels, labels[index]
            else:
                mixed_imgs = imgs
                lam = 1.0
                label_a, label_b = labels, labels

            # --- Forward Pass (AMP) ---
            with torch.amp.autocast('cuda', enabled=True):
                # Model outputs scaled logits
                logits = self.model(mixed_imgs)

                # Mixup Loss: Weighted sum of losses
                loss = lam * self.criterion(logits, label_a) + (1 - lam) * self.criterion(logits, label_b)

            # --- Backward Pass ---
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)

            # Gradient Clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            if hasattr(grad_norm, 'item'):
                current_norm = grad_norm.item()
            else:
                current_norm = float(grad_norm)
            max_grad_norm = max(max_grad_norm, current_norm)

            # Optimization Step
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()

            # --- Metrics & Logging ---
            loss_val = loss.item()
            total_loss += loss_val
            log_formatter.add(loss_val)

            # Accuracy Calculation (Approximate for Mixup)
            # We credit the prediction if it matches either label, weighted by lambda
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                if use_mixup:
                    acc_a = (preds == label_a).sum().item()
                    acc_b = (preds == label_b).sum().item()
                    # Weighted accuracy contribution
                    correct += (lam * acc_a + (1 - lam) * acc_b)
                else:
                    correct += (preds == labels).sum().item()
                total += labels.size(0)

            # Update Progress Bar
            if hasattr(iterator, 'set_postfix'):
                current_lr = self.optimizer.param_groups[0]['lr']
                iterator.set_postfix(loss=f'{loss_val:.4f}', lr=f'{current_lr:.4f}', grad=f'{current_norm:.2f}')

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0

        return (avg_loss, acc, max_grad_norm)