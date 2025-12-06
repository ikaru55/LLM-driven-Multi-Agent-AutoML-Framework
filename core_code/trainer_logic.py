import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import numpy as np
from torch.amp import autocast
from tqdm import tqdm

class CurricularFaceLoss(nn.Module):
    """
    CurricularFace Loss Implementation adapted for pre-computed logits.

    Paper: "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition"
    Strategy:
        - Dynamically adjusts the importance of negative samples.
        - Starts by focusing on easy samples, then gradually includes harder ones.
        - Prevents early training collapse (NaNs) better than ArcFace by managing gradient magnitude.

    Attributes:
        s (float): Scale factor (must match model's MetricHead s=64.0).
        m (float): Angular margin.
    """

    def __init__(self, s=64.0, m=0.5):
        super(CurricularFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.register_buffer('t', torch.zeros(1))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        cos_theta = logits / self.s
        cos_theta = cos_theta.clamp(-1.0 + 1e-07, 1.0 - 1e-07)
        target_cos = cos_theta[torch.arange(0, logits.size(0)), labels].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_cos, 2))
        cos_theta_m = target_cos * self.cos_m - sin_theta * self.sin_m
        if self.m > 0.0:
            cond_v = target_cos - self.threshold
            cond_mask = cond_v <= 0
            mm_tensor = torch.tensor(self.mm, dtype=target_cos.dtype, device=target_cos.device)
            keep_val = (target_cos - mm_tensor).to(cos_theta_m.dtype)
            cos_theta_m[cond_mask] = keep_val[cond_mask]
        with torch.no_grad():
            self.t = 0.99 * self.t + 0.01 * torch.mean(target_cos)
        mask = cos_theta > cos_theta_m
        final_cos = cos_theta.clone()
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        hard_negatives = final_cos[mask]
        t_tensor = self.t.to(hard_negatives.dtype).to(hard_negatives.device)
        noise = hard_negatives * (t_tensor + hard_negatives)
        final_cos[mask] = hard_negatives + noise
        output = one_hot * cos_theta_m + (1.0 - one_hot) * final_cos
        output = output * self.s
        return self.ce(output, labels)

class TrainerAdapter:
    """
    TrainerAdapter implementing the 'Recovery & Stabilization' strategy.

    Core Components:
        - Optimizer: AdamW (LR=1e-3, WD=5e-4) to fix initialization issues.
        - Scheduler: CosineAnnealingLR for smooth convergence.
        - Loss: CurricularFace to manage difficulty and prevent collapse.
        - Gradient Handling: Strict clipping and parameter grouping.
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.train_cfg = config.get('train', {})
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def setup_optimization(self, steps_per_epoch):
        """
        Sets up AdamW, Cosine Scheduler, and CurricularFace Loss.
        """
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith('.bias') or 'bn' in name or ('norm' in name):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        lr = 0.001
        weight_decay = 0.0005
        optim_groups = [{'params': decay_params, 'weight_decay': weight_decay}, {'params': no_decay_params, 'weight_decay': 0.0}]
        self.optimizer = optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08)
        epochs = int(self.train_cfg.get('epochs_full', 10))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-06)
        s_factor = 64.0
        if hasattr(self.model, 'head') and hasattr(self.model.head, 's'):
            s_factor = self.model.head.s
        self.criterion = CurricularFaceLoss(s=s_factor, m=0.5)
        self.criterion = self.criterion.to(self.device)

    def run_epoch(self, loader, epoch, scaler, log_formatter):
        """
        Standard Training Loop with Gradient Clipping and Logging.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        max_grad_norm = 0.0
        clip_grad = 5.0
        use_tty = sys.stdout.isatty()
        iterator = tqdm(loader, desc=f'Epoch {epoch}', leave=False, disable=not use_tty)
        self.optimizer.zero_grad()
        for i, (images, labels) in enumerate(iterator):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with autocast(device_type=self.device.type, enabled=True):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            max_grad_norm = max(max_grad_norm, float(grad_norm))
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
            current_loss = loss.item()
            running_loss += current_loss
            log_formatter.add(current_loss)
            with torch.no_grad():
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if use_tty:
                current_lr = self.optimizer.param_groups[0]['lr']
                iterator.set_postfix(loss=f'{current_loss:.4f}', acc=f'{100.0 * correct / max(1, total):.2f}%', lr=f'{current_lr:.5f}')
        self.scheduler.step()
        epoch_loss = running_loss / max(1, len(loader))
        epoch_acc = 100.0 * correct / max(1, total)
        return (epoch_loss, epoch_acc, max_grad_norm)