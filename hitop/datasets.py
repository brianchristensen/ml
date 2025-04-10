import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2 as T
from config import DatasetConfig

def get_dataset(cfg: DatasetConfig):
    if cfg.name.lower() == "cifar10":
        # Ensure every sample is a tensor image before GPU augments
        base_transform = T.ToImage()

        transform_train = cfg.transform_train or T.Compose([
            base_transform,
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.ToDtype(torch.float32, scale=True)
        ])
        transform_test = cfg.transform_test or T.Compose([
            base_transform,
            T.ToDtype(torch.float32, scale=True)
        ])

        train_data = datasets.CIFAR10(root=cfg.root, train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(root=cfg.root, train=False, download=True, transform=transform_test)

    else:
        raise ValueError(f"Dataset {cfg.name} not supported yet.")

    return train_data, test_data
