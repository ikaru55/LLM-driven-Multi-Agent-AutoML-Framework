import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional

class ArcFaceLoss(nn.Module):
    """
    Implements standard ArcFace Loss as requested by the Hacker.
    Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

    Why this over CurricularFace?
    - Stability: Pure geometric margin ($m=0.5$) is less prone to gradient explosions than curvature-adaptive losses.
    - Precision: Computations are forced to float32 where possible to prevent NaN during training.
    """
    def __init__(self, s=64.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Threshold for theta + m < pi
        self.threshold = math.cos(math.pi - m)
        # Fallback margin for hard samples
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        # logits are cos(theta) from the model's MetricHead
        # Ensure float32 for stability
        logits = logits.float() 

        # 1. Calculate cos(theta + m)
        # cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
        cos_theta = torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m

        # 2. Stability Condition
        # If cos_theta > threshold, use phi. Otherwise use linear penalty (cos_theta - mm)
        # This prevents gradients from exploding when theta + m approaches pi
        phi = torch.where(cos_theta > self.threshold, phi, cos_theta - self.mm)

        # 3. Apply Margin only to Positive Class
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        # Algebraic optimization:
        output = (one_hot * (phi - cos_theta)) + cos_theta

        # 4. Feature Re-scaling
        output *= self.s

        # 5. Cross Entropy
        return F.cross_entropy(output, labels)

class TrainerAdapter:
    """
    Refined Trainer Logic adhering to the 'Back to Basics' strategy.

    Key Changes:
    - Optimizer: SGD + Nesterov (Replaces AdamW for better angular convergence)
    - Scheduler: Linear Warmup (3 epochs) + Cosine Annealing
    - Loss: ArcFace (Standard)
    - Precision: Explicit Float32 handling in Loss and Gradient Clipping
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Placeholders
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training Hyperparameters (Strategy Overrides)
        self.lr = 0.1  # High initial LR for SGD
        self.weight_decay = 5e-4
        self.momentum = 0.9

        train_cfg = self.config.get('train', {})
        self.epochs = int(train_cfg.get('epochs_full', 10))

    def setup_optimization(self, steps_per_epoch):
        """
        Sets up SGD, Scheduler, and Loss based on the Agreed Strategy.
        """
        # 1. Parameter Grouping (Apply WD only to weights, not bias/BN)
        param_groups = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bn' in name or 'bias' in name:
                # No weight decay for Bias or BatchNorm parameters
                param_groups.append({'params': param, 'weight_decay': 0.0})
            else:
                param_groups.append({'params': param, 'weight_decay': self.weight_decay})

        # 2. Optimizer: SGD with Nesterov
        # Empirically superior for ArcFace compared to AdamW
        self.optimizer = optim.SGD(
            param_groups, 
            lr=self.lr, 
            momentum=self.momentum, 
            nesterov=True
        )

        # 3. Scheduler: Linear Warmup -> Cosine Annealing
        # Warmup helps the embeddings spread out on the hypersphere before margins kick in.
        warmup_epochs = 3
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear Warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine Annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # 4. Loss Function: Standard ArcFace
        # s=64.0, m=0.5 (Gold standard)
        self.criterion = ArcFaceLoss(s=64.0, m=0.50).to(self.device)

        print(f"[Trainer] Optimization Setup Complete:")
        print(f"  - Optimizer: SGD (lr={self.lr}, mom={self.momentum}, wd={self.weight_decay})")
        print(f"  - Scheduler: Linear Warmup ({warmup_epochs} eps) -> Cosine Annealing")
        print(f"  - Loss: ArcFace (s=64.0, m=0.50)")

    def run_epoch(self, loader, epoch, scaler, log_formatter):
        """
        Executes one training epoch with Gradient Clipping and AMP.
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        max_grad_norm = 0.0

        # Handle TQDM if available
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc=f"Ep {epoch}", leave=False)
        except ImportError:
            iterator = loader

        for imgs, labels in iterator:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Context
            # Note: Model head forces FP32 internally, but we keep autocast for backbone efficiency
            with torch.amp.autocast('cuda', enabled=True):
                # Forward
                cosine_logits = self.model(imgs)
                # Loss
                loss = self.criterion(cosine_logits, labels)

            # Backward with Scaler
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(self.optimizer)

            # Gradient Clipping (Value 5.0 as per strategy)
            # Essential for preventing explosions with PReLU/ArcFace
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

            # Safe norm extraction for logging
            if hasattr(grad_norm, 'item'):
                current_norm = grad_norm.item()
            else:
                current_norm = float(grad_norm)
            max_grad_norm = max(max_grad_norm, current_norm)

            # Optimizer Step
            scaler.step(self.optimizer)
            scaler.update()

            # Scheduler Step (Per-batch for smooth curve)
            self.scheduler.step()

            # Logging & Metrics
            loss_val = loss.item()
            total_loss += loss_val
            log_formatter.add(loss_val)

            with torch.no_grad():
                # Accuracy calc (based on ArcFace scaled logits or raw cosine)
                # Ideally check raw cosine, but scaled is monotonic w.r.t argmax
                preds = cosine_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            if hasattr(iterator, 'set_postfix'):
                current_lr = self.optimizer.param_groups[0]['lr']
                iterator.set_postfix(
                    loss=f"{loss_val:.4f}", 
                    lr=f"{current_lr:.4f}",
                    grad=f"{current_norm:.2f}"
                )

        # Epoch Aggregates
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, acc, max_grad_norm