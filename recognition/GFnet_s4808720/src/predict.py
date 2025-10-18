import torch
from dataset import get_dataloader

def main():
    """
    Main training entry point.
    """
    # hyperparameters
    batch_size = 8 # placeholder for now

    # Get DataLoaders
    train_loader, val_loader = get_dataloader(batch_size=batch_size, train=False)

if __name__ == "__main__":
    main()