import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

DATASET_PATH = "/home/groups/comp3710/ADNI/AD_NC"

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),   # Random zoom & crop
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),

    # apply small random rotation, translation, scaling, and shear together (one combined affine warp)
    transforms.RandomAffine(
        degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handles intensity differences between scanners
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)), # Gaussian noise injection
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)), # Random occlusion to improve robustness
    transforms.Normalize(mean=[0.5], std=[0.5])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class BrainMRIDataset(Dataset):
    """
    Custom Dataset for loading images from a directory structure.
    """
    def __init__(self, root_dir, train=True, transform=None):
        """
        Initializes the ADNIDataset.

        Args:
            root_dir (str or Path): Path to the root directory containing 'train' and 'test' folders.
            train (bool): If True, load training data. If False, load testing data.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = Path(root_dir) / ('train' if train else 'test')
        self.transform = transform
        self.samples = []

        # Loop over each class folder (NC, AD)
        for label, cls in enumerate(['NC', 'AD']):
            cls_path = self.root_dir / cls
            if not cls_path.exists():
                raise FileNotFoundError(f"Expected folder not found: {cls_path}")
            for fname in os.listdir(cls_path):
                img_path = cls_path / fname
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloader(batch_size, train=True, val_split=0.2):
    """
    Creates DataLoader for the BrainMRIDataset.

    Args:
        batchsize (int): Number of samples per batch.
        train (bool): If True, load training data. If False, load testing data.
        val_split (float): Proportion of training data to use for validation.
    Returns:
        If train is True:
            Tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.
        If train is False:
            DataLoader: A DataLoader for the test dataset.
    """
    if train:
        # load full training dataset
        full_dataset = BrainMRIDataset(DATASET_PATH, train=True, transform=TRAIN_TRANSFORM)

        # Split into training and validation subsets
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    
    else:
        # Test DataLoader
        test_dataset = BrainMRIDataset(root_dir=DATASET_PATH, train=False, transform=TEST_TRANSFORM)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return test_loader