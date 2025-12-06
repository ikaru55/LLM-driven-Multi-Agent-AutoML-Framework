import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import sys
import cv2  # [Speed Hack] OpenCV 추가
import numpy as np

try:
    from .dataset_logic import get_transforms
except ImportError:
    from dataset_logic import get_transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        if os.path.exists(root_dir):
            class_names = sorted(os.listdir(root_dir))
            for class_idx, class_name in enumerate(class_names):
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    # [Speed Hack] scandir이 listdir보다 파일 시스템 접근이 빠름
                    with os.scandir(class_path) as entries:
                        for entry in entries:
                            if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                self.image_paths.append(entry.path)
                                self.labels.append(class_idx)
            print(f"[{'Train' if is_train else 'Val'}] Loaded {len(self.image_paths)} images.")
        else:
            print(f"Warning: Path {root_dir} not found. Using dummy data.")
            for i in range(100):
                self.image_paths.append(f"dummy_{i}.jpg")
                self.labels.append(i % 10)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            if "dummy" in img_path:
                image = Image.new('RGB', (112, 112), color='red')
            else:
                # [Speed Hack] PIL.Image.open 대신 cv2 사용 (JPEG 디코딩 속도 2~3배 향상)
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    raise ValueError("Image None")
                # BGR -> RGB 변환 (cv2는 BGR, PIL/Torch는 RGB)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_cv) # Transform 호환성을 위해 PIL 객체로 변환 (비용 매우 적음)
        except Exception as e:
            # 에러 시 검은색 이미지 반환 (학습 중단 방지)
            image = Image.new('RGB', (112, 112), color='black')

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                import torchvision.transforms as T
                image = T.ToTensor()(image)

        return image, label

def get_dataloader(data_path, batch_size, is_train=True, num_workers=8, prefetch_factor=4, img_size=112):
    """
    [Hardware Optimization]
    Ryzen 7900X (12C/24T)를 위해 num_workers와 prefetch_factor를 상향 조정합니다.
    """
    # 7900X는 코어가 많으므로 worker를 8~12까지 늘려도 됩니다.
    # GPU가 데이터를 기다리지 않게 prefetch_factor를 높입니다.
    actual_workers = 8 if is_train else 4
    
    transform = get_transforms(is_train=is_train, img_size=img_size)
    dataset = FaceDataset(data_path, transform=transform, is_train=is_train)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=actual_workers,
        pin_memory=True, # GPU 전송 가속 필수
        persistent_workers=(actual_workers > 0), # [Speed Hack] 에포크마다 워커 재생성 방지
        prefetch_factor=prefetch_factor if actual_workers > 0 else None,
        drop_last=is_train,
    )
    return dataloader