import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

DATASET_PATH = "/home/groups/comp3710/ADNI/AD_NC"

class BrainMRIDataset(Dataset):
    """
    Custom Dataset for loading images from a directory structure.
    """
    def __init__(self, root_dir, train=True, transform=None):
        """
        Initializes the ANDIDataset.

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

# Example usage: 
dataset = BrainMRIDataset(DATASET_PATH, train=True, transform=None)
