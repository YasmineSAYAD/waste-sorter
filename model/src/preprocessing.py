import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────

IMAGE_SIZE = (224, 224)

# Calculated in eda.ipynb using the dataset 
MEAN_CUSTOM = [0.6493, 0.6345, 0.6164]
STD_CUSTOM  = [0.2574, 0.2545, 0.278]

# For MobileNet / YOLO pretrained ImageNet
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]

CLASSES = sorted([
    "battery", "cardboard", "electronic", "glass", "medical"
    "metal", "organic", "paper", "plastic", "textile", "trash",
])

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

NUM_CLASSES = len(CLASSES)


# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────

def get_transforms(mode: str, pretrained: bool = False) -> transforms.Compose:
    """
    Returns the transforms based on the mode and model type.

    Args:
        mode       : "train", "val" ou "test"
        pretrained : True for MobileNet/YOLO (ImageNet mean/std)
                     False for CNN scratch (mean/std custom)
    """
    mean = MEAN_IMAGENET if pretrained else MEAN_CUSTOM
    std  = STD_IMAGENET  if pretrained else STD_CUSTOM

    if mode == "train":
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class WasteDataset(Dataset):
    """
    Dataset PyTorch for waste sorter.
    load images from splits.json generated in eda.ipynb.
    """

    def __init__(
        self,
        samples: list[dict],
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        Args:
            samples   : liste de dicts {"path": str, "label": str}
            transform : transforms to apply
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        path   = sample["path"]
        label  = sample["label"]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, CLASS_TO_IDX[label]

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculates the class weights to compensate for the imbalance.
         Pass to nn.CrossEntropyLoss(weight=...).
        """
        counts = torch.zeros(NUM_CLASSES)
        for sample in self.samples:
            counts[CLASS_TO_IDX[sample["label"]]] += 1

        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * NUM_CLASSES
        return weights


# ─────────────────────────────────────────
# Dataloaders
# ─────────────────────────────────────────

def get_dataloaders(
    splits_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    pretrained: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Principal function — return train / val / test dataloaders.

    Args:
        splits_path : path to splits.json (generated in eda.ipynb)
        batch_size  : batch size
        num_workers : workers DataLoader (0 in Windows)
        pretrained  : True for MobileNet/YOLO, False for CNN scratch

    Returns:
        tuple (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     splits_path="model/data/splits/splits.json",
        ...     batch_size=32,
        ...     pretrained=False,
        ... )
    """
    splits_path = Path(splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(
            f"splits.json introuvable : {splits_path}\n"
            "Lance d'abord eda.ipynb pour générer les splits."
        )

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    train_dataset = WasteDataset(
        samples=splits["train"],
        transform=get_transforms("train", pretrained=pretrained),
    )
    val_dataset = WasteDataset(
        samples=splits["val"],
        transform=get_transforms("val", pretrained=pretrained),
    )
    test_dataset = WasteDataset(
        samples=splits["test"],
        transform=get_transforms("test", pretrained=pretrained),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train   : {len(train_dataset):>5} images — {len(train_loader):>4} batchs")
    print(f"Val     : {len(val_dataset):>5} images — {len(val_loader):>4} batchs")
    print(f"Test    : {len(test_dataset):>5} images — {len(test_loader):>4} batchs")
    print(f"Classes : {NUM_CLASSES} — {CLASSES}")
    print(f"Device  : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────

def denormalize(tensor: torch.Tensor, pretrained: bool = False) -> torch.Tensor:
    """
    Denormalizes an image tensor for matplotlib display.
    Usage : plt.imshow(denormalize(img_tensor).permute(1, 2, 0))
    """
    mean = torch.tensor(MEAN_IMAGENET if pretrained else MEAN_CUSTOM).view(3, 1, 1)
    std  = torch.tensor(STD_IMAGENET  if pretrained else STD_CUSTOM).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def get_device() -> torch.device:
    """Returns the available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")