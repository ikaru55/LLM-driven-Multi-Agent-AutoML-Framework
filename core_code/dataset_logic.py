import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
try:
    import cv2
except ImportError:
    cv2 = None

class CLAHE(object):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a PIL image.
    Strategy: Converts RGB to LAB, applies CLAHE to the L-channel (Luminance), 
    and converts back to RGB. This enhances local contrast without distorting color info.
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if cv2 is None:
            return img
        try:
            img_np = np.array(img)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            return Image.fromarray(final)
        except Exception:
            return img

    def __repr__(self):
        return f'{self.__class__.__name__}(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})'

class GridMask(object):
    """
    GridMask Augmentation.
    Strategy: Structured occlusion to force the model to learn distributed feature representations.
    Unlike RandomErasing, this preserves the structural context of the face.
    Operates on Tensors (C, H, W).
    """

    def __init__(self, d_range, ratio=0.5, prob=0.5):
        self.d_range = d_range
        self.ratio = ratio
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        c, h, w = img.size()
        d = np.random.randint(self.d_range[0], self.d_range[1])
        l = int(d * self.ratio)
        mask = np.ones((h, w), dtype=np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, h // d + 1):
            s_i = max(0, d * i + st_h)
            t_i = min(h, d * i + st_h + l)
            if s_i >= t_i:
                continue
            for j in range(-1, w // d + 1):
                s_j = max(0, d * j + st_w)
                t_j = min(w, d * j + st_w + l)
                if s_j >= t_j:
                    continue
                mask[s_i:t_i, s_j:t_j] = 0.0
        mask_tensor = torch.from_numpy(mask).to(img.dtype)
        mask_tensor = mask_tensor.expand_as(img)
        return img * mask_tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(d_range={self.d_range}, ratio={self.ratio}, prob={self.prob})'

def get_transforms(is_train: bool, img_size: int=256):
    """
    Defines the transform pipeline for training and validation.

    Strategy Implemented:
    - Target Resolution: 112x112 (Strict enforcement for MobileFaceNet efficiency).
    - Photometric Normalization: CLAHE (OpenCV) added to normalize local contrast and handle lighting variations.
    - Geometric Augmentation: RandomHorizontalFlip (Standard).
    - Structured Occlusion: GridMask added to replace RandomErasing. Prevents overfitting to specific facial landmarks 
      by forcing the network to use partial information.
    - Mixup: VETOED for this cycle to prioritize stability after previous collapse.
    """
    target_size = 112
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if is_train:
        transform = transforms.Compose([transforms.Resize((target_size, target_size)), CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), GridMask(d_range=(20, 40), ratio=0.5, prob=0.5)])
    else:
        transform = transforms.Compose([transforms.Resize((target_size, target_size)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transform