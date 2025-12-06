import torch
import torch.nn as nn
import time
import os
import sys
import numpy as np
import yaml
import torch.nn.functional as F
import json
from tqdm import tqdm
from sklearn.metrics import roc_curve
from PIL import Image
import torchvision.transforms as transforms
from copy import deepcopy
import traceback
import gc
import cv2

# --- Imports ---
try:
    from .dataset import get_dataloader
    from .model import get_model
    from .trainer_logic import TrainerAdapter
except ImportError:
    from dataset import get_dataloader
    from model import get_model
    from trainer_logic import TrainerAdapter

# --- Logging Helper ---
class AdvancedLogFormatter:
    def __init__(self):
        self.losses = []
    
    def add(self, loss_val: float):
        self.losses.append(float(loss_val))
    
    def _generate_sparkline(self, data, width=30):
        if not data: return ""
        
        # [Fix] Filter out NaN and Inf values
        clean_data = [x for x in data if np.isfinite(x)]
        if not clean_data: return "⚠" * min(width, len(data))  # All NaN/Inf
        
        n = len(clean_data)
        if n > width:
            idx = np.linspace(0, n, width + 1, dtype=int)
            sampled = [np.nanmean(clean_data[idx[i]:idx[i + 1]]) for i in range(width)]
        else:
            sampled = clean_data
        
        # [Fix] Additional NaN check after sampling
        sampled = [x if np.isfinite(x) else 0.0 for x in sampled]
        if not sampled: return "─"
        
        lo, hi = min(sampled), max(sampled)
        if hi == lo: return "─" * len(sampled)
        
        normalized = [(x - lo) / (hi - lo) for x in sampled]
        bars = "  ▂▃▄▅▆▇█"
        return "".join(bars[min(len(bars)-1, max(0, int(val * (len(bars) - 1))))] for val in normalized)
    
    def get_detailed_report(self, epoch: int) -> str:
        if not self.losses: return f"Epoch {epoch} | No Data"
        data = np.array(self.losses, dtype=float)
        
        # [Fix] Handle NaN values in statistics
        clean_data = data[np.isfinite(data)]
        nan_count = len(data) - len(clean_data)
        
        sparkline = self._generate_sparkline(data.tolist(), width=30)
        
        if len(clean_data) > 0:
            mean = float(np.mean(clean_data))
            std = float(np.std(clean_data))
            max_val = float(np.max(clean_data))
            min_val = float(np.min(clean_data))
        else:
            mean = std = max_val = min_val = float('nan')
        
        report = (f"Epoch {epoch} Report:\n"
                  f"  • Trend : [{sparkline}] (Left=Start, Right=End)\n"
                  f"  • Stats : Avg={mean:.4f}, Std={std:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")
        
        if nan_count > 0:
            report += f"\n  ⚠️ WARNING: {nan_count} NaN/Inf loss values detected!"
        
        return report

# --- Validation Logic ---
class ValidationListDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, root, transform):
        self.files = file_list
        self.root = root
        self.transform = transform
    
    def __len__(self): return len(self.files)
    
    def __getitem__(self, i):
        # [Path Fix] 경로 호환성 처리
        root_cleaned = self.root.rstrip(os.sep).rstrip('/')
        file_cleaned = self.files[i].lstrip(os.sep).lstrip('/')
        path = os.path.join(root_cleaned, file_cleaned)
        path = path.replace('/', os.sep).replace('\\', os.sep)
        
        if not os.path.exists(path):
            # 검증 데이터 누락 방지 (검은색 이미지 반환)
            print(f"❌ [Val] Image not found: {path}")
            return torch.zeros((3, 112, 112)), self.files[i]

        try:
            # [Speed Hack] Validation에서도 OpenCV 사용 (Linux에서 더욱 효율적)
            img_cv = cv2.imread(path)
            if img_cv is None: raise Exception("Img None")
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_cv)
        except Exception as e:
            print(f"❌ Error loading image {path}: {e}")
            image = Image.new('RGB', (112, 112), color='black')
            
        return self.transform(image), self.files[i]

def validate_verification(model, test_list_path, test_path, device):
    model.eval()
    if not os.path.exists(test_list_path):
        print(f"[Validate] ❌ Test list not found: {test_list_path}")
        return 50.0

    with open(test_list_path, 'r') as f: lines = f.readlines()
    files = sum([x.strip().split(',')[-2:] for x in lines], [])
    setfiles = sorted(list(set(files)))
    
    valid_files = []
    
    # 경로 미리 검사
    for fname in setfiles:
        fpath = os.path.join(test_path, fname).replace('/', os.sep).replace('\\', os.sep)
        if os.path.exists(fpath):
            valid_files.append(fname)
    
    print(f"[Validate] Loaded {len(lines)} pairs, {len(valid_files)}/{len(setfiles)} valid images.")

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_ds = ValidationListDataset(valid_files, test_path, transform)
    
    # [Linux Optimization] Worker 수 자동 조절
    num_cpus = os.cpu_count()
    val_workers = min(8, num_cpus) if num_cpus else 4
    
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=256, 
        shuffle=False, 
        num_workers=val_workers, 
        pin_memory=True,
        persistent_workers=(val_workers > 0) # Linux에서 프로세스 재생성 오버헤드 방지
    )

    feature_dict = {}
    
    with torch.no_grad():
        for imgs, fnames in tqdm(val_loader, desc="Validating", leave=False):
            # 메모리 포맷을 모델과 맞춤 (Channels Last)
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            feats = model(imgs)
            feats = F.normalize(feats, p=2, dim=1)
            feats = feats.cpu().numpy()
            for i, fname in enumerate(fnames):
                feature_dict[fname] = feats[i]

    all_scores = []
    all_labels = []
    
    for line in lines:
        parts = line.strip().split(',')
        label = int(parts[0])
        f1, f2 = parts[1], parts[2]
        
        if f1 in feature_dict and f2 in feature_dict:
            score = np.dot(feature_dict[f1], feature_dict[f2])
            all_scores.append(score)
            all_labels.append(label)

    if len(all_scores) == 0: 
        print("[Validate] ❌ No scores calculated. Returning default 50.0")
        return 50.0

    # [Fix] Handle NaN scores properly
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    nan_mask = ~np.isfinite(all_scores)
    nan_count = np.sum(nan_mask)
    
    if nan_count > 0:
        nan_ratio = nan_count / len(all_scores)
        print(f"[Validate] ⚠️ {nan_count} NaN scores detected ({nan_ratio*100:.1f}%)")
        
        # NaN이 50% 이상이면 validation 결과를 신뢰할 수 없음
        if nan_ratio > 0.5:
            print(f"[Validate] ❌ Too many NaN scores ({nan_ratio*100:.1f}%). Model weights corrupted. Returning 50.0")
            return 50.0
        
        # NaN을 제거하고 유효한 score만 사용
        valid_mask = np.isfinite(all_scores)
        all_scores = all_scores[valid_mask]
        all_labels = all_labels[valid_mask]
    
    if len(all_scores) == 0:
        print("[Validate] ❌ No valid scores after NaN filtering. Returning 50.0")
        return 50.0

    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[idx] * 100
    return eer

# --- Early Stopper ---
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# --- Main Execution ---
def main(base_config):
    config = base_config
    train_cfg = config.get('train', {})
    data_cfg = config.get('data', {})

    # =================================================================
    # [Speed Hack] NVIDIA RTX 30/40/50 Series & Linux Optimization Logic
    # =================================================================
    if torch.cuda.is_available():
        # 1. Benchmark: 고정된 입력 크기(112x112)에서 최적의 알고리즘 탐색
        torch.backends.cudnn.benchmark = True
        
        # 2. TF32 (TensorFloat-32): Ampere/Ada 아키텍처 핵심 가속 기능
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("[Trainer] 🚀 NVIDIA RTX Optimization: CuDNN Benchmark & TF32 Enabled.")
    # =================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [OOM Fix] 초기화 전 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        print("[Trainer] 🧹 GPU Cache Cleared before model loading.")

    mode = config.get('mode', 'full')
    is_test = config.get('test_mode', False)

    if mode == 'fast' or is_test:
        fast_epochs = int(train_cfg.get('epochs_fast', 1))
        train_cfg['epochs_full'] = fast_epochs
        epochs = fast_epochs
    else:
        epochs = int(train_cfg.get('epochs_full', 10))

    es_patience = int(train_cfg.get('early_stop_patience', 5))
    es_delta = float(train_cfg.get('early_stop_delta', 0.0))
    early_stopper = EarlyStopper(patience=es_patience, min_delta=es_delta)

    print("[Trainer] Loading Data...")
    
    # [Linux Optimization] CPU Core에 맞춰 Worker 자동 할당
    # Linux는 Fork 방식을 사용하여 Worker 생성 비용이 적고 병렬 처리가 강력함
    num_cpus = os.cpu_count()
    # 너무 많은 worker는 오히려 오버헤드 발생 가능 (보통 8~16 권장)
    optimal_workers = min(12, num_cpus) if num_cpus else 8
    print(f"[Trainer] 🐧 Linux Optimization: Setting num_workers to {optimal_workers}")

    train_loader = get_dataloader(
        data_path=data_cfg.get('train1_path'), 
        batch_size=train_cfg.get('batch_size', 64), 
        is_train=True, 
        num_workers=optimal_workers,    
        prefetch_factor=4,   
        img_size=data_cfg.get('img_size', 112)
    )

    # Model Setup
    train_path = data_cfg.get('train1_path', '')
    if 'num_classes' in data_cfg: num_classes = int(data_cfg['num_classes'])
    elif 'train2' in train_path: num_classes = 949
    elif 'train1' in train_path: num_classes = 2882
    else: num_classes = 2882

    img_size = data_cfg.get('img_size', 112)
    try: model = get_model(num_classes=num_classes, img_size=img_size)
    except: model = get_model(num_classes=num_classes)
    
    # Pretrained Loading Logic
    pretrained_path = train_cfg.get('pretrained_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[Trainer] 🔄 Loading Pre-trained Weights from {pretrained_path}...")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            model_state = model.state_dict()
            loaded_state = {}
            for k, v in checkpoint.items():
                if k in model_state:
                    if v.size() == model_state[k].size():
                        loaded_state[k] = v
            model.load_state_dict(loaded_state, strict=False)
            print("[Trainer] ✅ Pre-trained weights loaded successfully.")
        except Exception as e:
            print(f"[Trainer] ❌ Failed to load pre-trained weights: {e}")

    model = model.to(device)
    
    # [Speed Hack] Channels Last Memory Format
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("[Trainer] ⚡ Model converted to Channels Last format.")

    # =================================================================
    # [Linux Exclusive Speed Hack] Torch Compile
    # =================================================================
    # PyTorch 2.x의 컴파일 기능을 사용하여 그래프 최적화 및 커널 퓨전 수행
    # Linux 환경에서 가장 안정적이고 빠른 속도를 제공함
    if sys.platform.startswith('linux') and hasattr(torch, 'compile'):
        print("[Trainer] 🐧 Linux Detected: Applying torch.compile()...")
        try:
            # mode='reduce-overhead': 작은 배치나 반복 루프 오버헤드 감소에 최적
            # mode='max-autotune': 속도는 가장 빠르지만 컴파일 시간이 오래 걸림
            model = torch.compile(model, mode='reduce-overhead')
            print("[Trainer] 🚀 Model compiled successfully with 'reduce-overhead' mode.")
        except Exception as e:
            print(f"[Trainer] ⚠️ torch.compile failed (proceeding without it): {e}")
    # =================================================================

    # Adapter Setup
    print("🔧 Initializing Trainer Logic Adapter...")
    try:
        trainer_logic = TrainerAdapter(model, config, device)
        trainer_logic.setup_optimization(steps_per_epoch=len(train_loader))
    except Exception as e:
        print(f"❌ Error in TrainerAdapter setup: {e}")
        traceback.print_exc()
        return

    # GradScaler - 기본 설정 사용 (자동으로 scale 조정)
    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.get('use_amp', True))
    
    best_eer = 100.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        formatter = AdvancedLogFormatter()
        start_time = time.time()
        
        try:
            # Train Epoch
            epoch_loss, train_acc, grad_norm = trainer_logic.run_epoch(
                loader=train_loader, 
                epoch=epoch, 
                scaler=scaler, 
                log_formatter=formatter
            )
        except Exception as e:
            print(f"❌ Error during training epoch: {e}")
            traceback.print_exc()
            break
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"[Metrics] Speed: {epoch_duration:.2f}s/epoch")
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Grad: {grad_norm:.4f}")
        print(formatter.get_detailed_report(epoch))
        
        # Validation
        if True:
            val_eer = validate_verification(model, data_cfg.get('test_list'), data_cfg.get('val_path'), device)
            print(f"Val EER: {val_eer:.2f}%")
            
            # Save Best Model
            if val_eer < best_eer:
                best_eer = val_eer
                best_epoch = epoch
                print(f"[Metrics] Best EER Updated: {best_eer:.2f}% at Epoch {best_epoch}")
                try:
                    save_dir = "core_code"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "trained_model.pth")
                    # torch.compile된 모델은 state_dict 저장 시 prefix 문제가 있을 수 있어 원본 저장 권장
                    # 하지만 일반적인 save/load는 지원함
                    if hasattr(model, '_orig_mod'): # compiled model handling
                        torch.save(model._orig_mod.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                    print(f"[Trainer] 💾 Best model saved to {save_path}")
                except Exception as e:
                    print(f"[Trainer] ⚠️ Failed to save best model: {e}")

            # Save Checkpoint (Every Epoch)
            ckpt_dir = train_cfg.get('checkpoint_dir', os.path.join("core_code", "checkpoints"))
            os.makedirs(ckpt_dir, exist_ok=True)
            epoch_save_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth")
            try:
                if hasattr(model, '_orig_mod'):
                    torch.save(model._orig_mod.state_dict(), epoch_save_path)
                else:
                    torch.save(model.state_dict(), epoch_save_path)
            except: pass 

            if early_stopper.early_stop(val_eer):
                print(f"[Trainer] 🛑 Early Stopping triggered at epoch {epoch}")
                break

    print(f"[Result] Best Epoch: {best_epoch}")
    print(f"[Result] Best EER: {best_eer:.2f}")

if __name__ == "__main__":
    if os.path.exists("base_config.yaml"):
        with open("base_config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"train": {}, "data": {}}
    main(config)