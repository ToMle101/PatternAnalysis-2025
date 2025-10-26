import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import argparse
from dataset import get_dataloader
from modules import GFNetAlzheimers

def main():
    """
    Main training entry point.
    """

    # adjustable hyperparameters
    parser = argparse.ArgumentParser(description="Train GFNetAlzheimers on ADNI dataset")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout rate in model (default: 0.1)")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="DropPath rate (default: 0.1)")

    args = parser.parse_args()

    #device set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader = get_dataloader(batch_size=args.batch_size, train=True)

    # model set up
    model = GFNetAlzheimers(
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=2,
        embed_dim=128,
        depth=10,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        use_attention=True
    ).to(device)

if __name__ == "__main__":
    main()




