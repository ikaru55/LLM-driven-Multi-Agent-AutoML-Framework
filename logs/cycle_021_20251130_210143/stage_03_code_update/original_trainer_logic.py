import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple

class ElasticFaceLoss(nn.Module):
    """
    Implements ElasticFace (Arc-based) Loss with Label Smoothing.

    Strategy:
    - Randomized Margin: m ~ N(mean, std) to prevent overfitting to specific boundaries.
    - Scale (s): 64.0 (Standard for face recognition).
    - Label Smoothing: Integrated into CrossEntropy to prevent logit explosion.

    References:
    - ElasticFace: Elastic Margin Loss for Deep Face Recognition (CVPR 2022)
    """
    def __init__(self, s=64.0, m=0.5, std=0.05, label_smoothing=0.1):
        super(ElasticFaceLoss, self).__init__()
        self.s = s
        self.m_mean = m
        self.m_std = std
        self.label_smoothing = label_smoothing

        # Cache basic logical margins for reference, though we compute dynamically
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # For numerical stability
        self.epsilon = 1e-6

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Cosine similarity from the model head (B, NumClasses). 
                    Expected to be already normalized if coming from MetricHead.
            labels: Ground truth labels (B).
        """
        # Ensure float32 for metric learning precision
        cosine = logits.float()

        # 1. Clamp for numerical stability in acos
        cosine = torch.clamp(cosine, -1.0 + self.epsilon, 1.0 - self.epsilon)

        # 2. Sample random margin 'm' for this batch
        # ElasticFace: Sample m per sample or per batch. Per sample is more robust.
        # m ~ N(0.5, 0.05)
        with torch.no_grad():
            m_random = torch.normal(mean=self.m_mean, std=self.m_std, size=labels.size(), device=cosine.device)
            m_random = torch.clamp(m_random, min=0.0) # Margin shouldn't be negative practically

        # 3. Calculate theta
        acos = torch.acos(cosine)

        # 4. Apply margin to target classes
        # Target: cos(theta + m)
        target_cosine = torch.cos(acos + m_random.unsqueeze(1))

        # 5. Construct final logits
        # We only apply the margin modification to the specific Ground Truth index
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # logic: output = one_hot * target_cosine + (1 - one_hot) * cosine
        # But efficiently: output = cosine + one_hot * (target_cosine - cosine)
        # Note: We must ensure we pick the specific target_cosine corresponding to the label

        # Gather the specific target cosine values for the indices
        # Since target_cosine is (B, C) computed with broadcasting m, it's correct.
        # However, to be computationally efficient and correct with the tensor shape:
        # acos is (B, C). m_random is (B). 
        # We only strictly need to modify the diagonal (batch_idx, label).

        # Efficient approach:
        # Get cos(theta_y)
        # Calculate cos(theta_y + m_i)
        # Replace in logits

        # Gather cos(theta) for the ground truth
        index = labels.view(-1, 1).long()
        cos_theta_y = torch.gather(cosine, 1, index)
        theta_y = torch.acos(cos_theta_y)

        # Calculate angular margin
        cos_theta_y_m = torch.cos(theta_y + m_random.view(-1, 1))

        # Replace
        final_logits = cosine + one_hot * (cos_theta_y_m - cos_theta_y)

        # 6. Scale
        scaled_logits = final_logits * self.s

        # 7. Cross Entropy with Label Smoothing
        loss = F.cross_entropy(scaled_logits, labels, label_smoothing=self.label_smoothing)

        return loss

class TrainerAdapter:
    """
    Trainer Adapter implementing the "Hacker's Refined Strategy".

    Components:
    - Optimizer: SGD (Momentum 0.9, Nesterov)
    - Scheduler: Cosine Annealing (Per-iteration step, 0.1 -> 0.0)
    - Loss: ElasticFace-Arc (Randomized Margin)
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Strategy Hyperparameters
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 5e-4

        # Placeholders
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        train_cfg = self.config.get('train', {})
        self.epochs = int(train_cfg.get('epochs_full', 10))

    def setup_optimization(self, steps_per_epoch):
        """
        Setup Optimization Engine.
        """
        # 1. Parameter Groups (Weight Decay Decoupling)
        # Exclude bias and BatchNorm from weight decay for better stability
        param_groups = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'bn' in name or 'bias' in name:
                param_groups.append({'params': param, 'weight_decay': 0.0})
            else:
                param_groups.append({'params': param, 'weight_decay': self.weight_decay})

        # 2. Optimizer: SGD with Nesterov
        self.optimizer = optim.SGD(
            param_groups, 
            lr=self.lr, 
            momentum=self.momentum, 
            nesterov=True
        )

        # 3. Scheduler: Cosine Annealing (Per-iteration)
        # Hacker's spec: "Smooth decay from 0.1 -> 0.0 over the full epoch count"
        # We step this every batch to achieve the smoothest curve.
        total_steps = self.epochs * steps_per_epoch
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=0.0
        )

        # 4. Loss: ElasticFace
        self.criterion = ElasticFaceLoss(
            s=64.0, 
            m=0.5, 
            std=0.05, 
            label_smoothing=0.1
        ).to(self.device)

        print(f"[Trainer] Strategy Applied:")
        print(f"  • Optimizer : SGD (lr={self.lr}, mom={self.momentum}, wd={self.weight_decay}, nesterov=True)")
        print(f"  • Scheduler : CosineAnnealingLR (per-step, T_max={total_steps})")
        print(f"  • Criterion : ElasticFace (s=64.0, m~N(0.5, 0.05), LS=0.1)")

    def run_epoch(self, loader, epoch, scaler, log_formatter) -> Tuple[float, float, float]:
        """
        Runs one training epoch.
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        max_grad_norm = 0.0

        # Setup Progress Bar
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc=f"Ep {epoch}", leave=False)
        except ImportError:
            iterator = loader

        for imgs, labels in iterator:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Forward
            with torch.amp.autocast('cuda', enabled=True):
                # Model returns raw cosine similarities
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

            # Backward
            scaler.scale(loss).backward()

            # Unscale for gradient clipping
            scaler.unscale_(self.optimizer)

            # Gradient Clipping (Threshold 5.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            if hasattr(grad_norm, 'item'):
                current_norm = grad_norm.item()
            else:
                current_norm = float(grad_norm)
            max_grad_norm = max(max_grad_norm, current_norm)

            # Optimizer Step
            scaler.step(self.optimizer)
            scaler.update()

            # Scheduler Step (Per-iteration)
            self.scheduler.step()

            # Logging & Metrics
            loss_val = loss.item()
            total_loss += loss_val
            log_formatter.add(loss_val)

            with torch.no_grad():
                # For accuracy, we just look at the raw cosine scores
                # No need to apply margin for inference/accuracy check
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            if hasattr(iterator, 'set_postfix'):
                current_lr = self.optimizer.param_groups[0]['lr']
                iterator.set_postfix(
                    loss=f"{loss_val:.4f}", 
                    lr=f"{current_lr:.4f}", 
                    grad=f"{current_norm:.2f}"
                )

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        acc = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, acc, max_grad_norm