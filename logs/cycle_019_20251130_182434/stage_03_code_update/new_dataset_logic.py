import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import random

# [Strategy Implementation] Custom GridMask Module
class GridMask(nn.Module):
    """
    Implements GridMask augmentation for structured information deletion.
    Unlike RandomErasing which removes random rectangles, GridMask hides information
    in a structured grid pattern, forcing the network to learn distributed representations
    (holistic face understanding) rather than overfitting to specific landmarks.

    References:
    - GridMask Data Augmentation (arXiv:2001.04086)
    """
    def __init__(self, p=0.5, d_min=20, d_max=40, r_min=0.4, r_max=0.7):
        """
        Args:
            p (float): Probability of applying GridMask.
            d_min, d_max (int): Range for the grid unit size (d).
            r_min, r_max (float): Range for the keep ratio (r). 
                                  Note: r is the ratio of the retained region.
        """
        super().__init__()
        self.p = p
        self.d_min = d_min
        self.d_max = d_max
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, img):
        """
        Args:
            img (torch.Tensor): Image tensor of shape (C, H, W).
                                Assumed to be normalized to [-1, 1] (mean=0.5).
        Returns:
            torch.Tensor: GridMasked image.
        """
        if random.random() > self.p:
            return img

        c, h, w = img.size()

        # 1. Grid Parameters
        # d: Grid unit size
        d = random.randint(self.d_min, self.d_max)
        # r: Ratio of the "keep" region side length to unit size
        r = random.uniform(self.r_min, self.r_max)

        # l: Length of the square to KEEP
        l = int(d * r)

        # 2. Create Mask
        # Create a mask larger than the image to handle random offsets
        mask_h = h + d
        mask_w = w + d

        # Generate coordinate grids
        y = torch.arange(mask_h).view(-1, 1)
        x = torch.arange(mask_w).view(1, -1)

        # Logic: 1 if within the 'keep' region, 0 otherwise (Occlusion)
        # The region (x % d < l) AND (y % d < l) is kept.
        mask_y = (y % d < l).float()
        mask_x = (x % d < l).float()

        # Combined mask (intersection of x and y keep regions)
        mask = mask_y * mask_x

        # 3. Random Offset (Grid Placement)
        st_h = random.randint(0, d - 1)
        st_w = random.randint(0, d - 1)

        # Crop mask to match image size
        mask = mask[st_h:st_h+h, st_w:st_w+w]

        # 4. Apply to Image
        # Expand mask to channel dimension: (H, W) -> (C, H, W)
        mask = mask.expand_as(img)

        # Apply structured dropout:
        # Where mask is 1, keep image. Where mask is 0, fill with -1.0 (Black).
        # Note: Since normalization is mean=0.5, std=0.5, pixel 0 maps to -1.0.
        img = img * mask + (-1.0) * (1.0 - mask)

        return img

def get_transforms(is_train: bool, img_size: int=256):
    """
    Implements the "Hacker's Refined Strategy" for data augmentation.

    Strategy Overview:
    1. **Geometric Robustness:** - `Resize(128)` -> `RandomCrop(112)`: Implicit scale variation.
       - `RandomAffine`: Explicit rotation (+/- 10) and translation (+/- 0.1).
       - **Constraint:** No Shear (to preserve facial geometry).
    2. **Photometric Regularization:**
       - `ColorJitter`: Light intensity/contrast adaptation.
    3. **Structural Regularization:**
       - `GridMask`: Replaces RandomErasing to force distributed feature learning.
    4. **Normalization:** - Strict mapping to [-1, 1] via mean/std [0.5, 0.5, 0.5].

    Args:
        is_train (bool): Train vs Eval mode.
        img_size (int): Hint, but target is fixed to 112x112 for GhostNet.
    """
    target_size = 112
    # Resize slightly larger to allow for RandomCrop (implicit scale augmentation)
    resize_size = 128

    # Standard Face Verification Normalization
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if is_train:
        transform = transforms.Compose([
            # 1. Geometric Variation
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),

            # Hacker's Constraint: Rot +/- 10, Trans 0.1, No Shear
            # This exposes the model to the affine manifold required for head pose robustness.
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0),

            # 2. Photometric Variation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),

            # 3. Tensor Conversion & Normalization
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),

            # 4. Structural Regularization (GridMask)
            # Applied on the normalized tensor.
            GridMask(p=0.5, d_min=20, d_max=40, r_min=0.4, r_max=0.7)
        ])
    else:
        # Validation: Deterministic Pipeline
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transform