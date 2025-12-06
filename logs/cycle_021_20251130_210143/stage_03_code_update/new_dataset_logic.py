import torch
import torchvision.transforms as transforms

# [NOTE] GridMask removed as per "Hacker's Refined Strategy" to prevent 
# deterministic overfitting. Replaced by TrivialAugmentWide.

def get_transforms(is_train: bool, img_size: int = 256):
    """
    Implements the "Hacker's Refined Strategy" for data augmentation.

    Strategy Overview:
    1. **Stochastic Augmentation Policy (Train):**
       - `Resize(128)` -> `RandomCrop(112)`: Provides basic scale and shift invariance.
       - `TrivialAugmentWide`: A parameter-free, state-of-the-art automatic augmentation method.
         It samples a single augmentation (Geometric or Photometric) with a random magnitude
         from a vast search space. This maximizes sample diversity and prevents the model 
         from memorizing specific augmentation patterns (like fixed ColorJitter noise).
       - `RandomHorizontalFlip`: Essential symmetry for face verification.

    2. **Validation Pipeline (Test):**
       - Strict deterministic resizing to 112x112.
       - No test-time augmentation (TTA) to ensure consistent metric evaluation.

    3. **Normalization:**
       - Maps inputs to [-1, 1] via mean/std [0.5, 0.5, 0.5].

    Args:
        is_train (bool): Train vs Eval mode.
        img_size (int): Hint, but target is explicitly fixed to 112x112 for the GhostNet backbone.
    """

    # [CONSTRAINT] GhostNet 1.0x input size is fixed to 112
    target_size = 112
    # Resize slightly larger to allow for random cropping (Scale Invariance)
    resize_size = 128

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if is_train:
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),

            # [STRATEGY] TrivialAugmentWide
            # Injected as per Hacker's plan to replace manual chains (ColorJitter/Affine/GridMask).
            # This ensures the model sees a "new" version of the face almost every epoch.
            transforms.TrivialAugmentWide(),

            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # [VALIDATION] Standard deterministic pipeline
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transform