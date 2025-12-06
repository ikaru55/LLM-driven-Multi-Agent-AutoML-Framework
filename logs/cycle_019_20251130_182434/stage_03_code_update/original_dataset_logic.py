import torch
import torchvision.transforms as transforms

def get_transforms(is_train: bool, img_size: int = 256):
    """
    Implements the "Deterministic Geometry" Protocol for data augmentation as per the 
    Hacker's Refined Strategy.

    Strategy Implementation:
    1. **Geometric Determinism:** Replaces chaotic TrivialAugment/RandomPerspective with 
       a standard `Resize(128) -> RandomCrop(112)` pipeline. This simulates scale and 
       translation changes necessary for robust face verification without destroying 
       facial geometry (eyes, nose alignment).
    2. **Photometric Regularization:** Uses low-intensity `ColorJitter` to handle 
       lighting variations (brightness/contrast) which are common in face verification.
    3. **Information Deletion:** Retains `RandomErasing` (p=0.2) to force the model 
       to learn holistic features rather than relying on specific facial landmarks.
    4. **Normalization:** Strictly maps to [-1, 1] using mean/std [0.5, 0.5, 0.5].

    Args:
        is_train (bool): Whether to apply training augmentation.
        img_size (int): Hint for image size. Logic internally overrides this to 112 
                        to ensure compatibility with the GhostNet architecture.
    """
    # Strategy mandates 112x112 input resolution
    target_size = 112
    # Resize slightly larger for RandomCrop (Scale/Translation simulation)
    resize_size = 128

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if is_train:
        transform = transforms.Compose([
            # 1. Deterministic Geometry: Resize larger -> Random Crop
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop((target_size, target_size)),

            # 2. Standard Flip
            transforms.RandomHorizontalFlip(p=0.5),

            # 3. Low-Intensity Photometric Noise
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),

            # 4. Conversion & Normalization
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),

            # 5. Regularization (after normalization)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
        ])
    else:
        # Validation: Strict, Deterministic Resize to Target
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transform