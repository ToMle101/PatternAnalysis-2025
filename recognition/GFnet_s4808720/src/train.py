import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse
from dataset import get_dataloader
from modules import GFNetAlzheimers

# helper functions

def train_single_epoch(epoch, model, train_loader, criterion, optimizer, scheduler,
                    train_losses, train_accuracies, device):
    """Train the model for one epoch. """
    model.train()  # set model to training mode
    running_loss, correct, total = 0.0, 0, 0

    # loop through all training batches
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # reset gradients
        outputs = model(images)         # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()                 # backpropagate gradients
        optimizer.step()                # update model weights

        # track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # save average loss and accuracy for this epoch
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)


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

    # loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # metric tracking
    # Lists to store loss and accuracy values over epochs for plotting later
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # training loop
    for epoch in range(args.epochs):
        train_single_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, train_losses, train_accuracies, device)


if __name__ == "__main__":
    main()




