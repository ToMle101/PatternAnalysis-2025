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
    """
    Train the model for a single epoch.

    Args:
        epoch (int): Current epoch index.
        model (nn.Module): The model being trained.
        train_loader (DataLoader): Dataloader providing batches of training data.
        criterion (torch.nn.Module): Loss function used to compute training loss.
        optimizer (torch.optim.Optimizer): Optimizer responsible for updating model parameters.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        train_losses (list): List to store average training loss for each epoch.
        train_accuracies (list): List to store training accuracy for each epoch.
        device (torch.device): Device to run computations on ('cuda' or 'cpu').

    Returns:
        None. Updates `train_losses` and `train_accuracies` in-place. 
    """
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


def validate_single_epoch(epoch, model, val_loader, criterion,
                    val_losses, val_accuracies, device):
    """
    Evaluate the model on the validation dataset for one epoch.

    Args:
        epoch (int): Current epoch index.
        model (nn.Module): The model being evaluated.
        val_loader (DataLoader): Dataloader providing validation batches.
        criterion (torch.nn.Module): Loss function used to compute validation loss.
        val_losses (list): List to store average validation loss per epoch.
        val_accuracies (list): List to store validation accuracy per epoch.
        device (torch.device): Device to run computations on ('cuda' or 'cpu').

    Returns:
        None. Updates `val_losses` and `val_accuracies` in-place.
    """
    model.eval()  
    running_loss, correct, total = 0.0, 0, 0

    # Disable gradient calculations for validation (faster, uses less memory)
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Save average loss and accuracy for this epoch
    val_losses.append(running_loss / len(val_loader))
    val_accuracies.append(100 * correct / total)


def plot_stats(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot and save training and validation loss/accuracy curves.

    Args:
        num_epochs (int): Total number of epochs trained.
        train_losses (list): Recorded training losses.
        val_losses (list): Recorded validation losses.
        train_accuracies (list): Recorded training accuracies.
        val_accuracies (list): Recorded validation accuracies.

    Returns:
        None. Saves plots as 'training_metrics.png'.
    """
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label="Train Accuracy")
    plt.plot(range(num_epochs), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    print("Saved training plots as training_metrics.png")

def main():
    """
    Main training entry point.
    """

    # adjustable hyperparameters
    parser = argparse.ArgumentParser(description="Train GFNetAlzheimers on ADNI dataset")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (default: 1e-5)")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout rate in model (default: 0.1)")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="DropPath rate (default: 0.1)")

    #parse the arguments from the command line
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
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    # training loop
    for epoch in range(args.epochs):
        train_single_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, train_losses, train_accuracies, device)
        validate_single_epoch(epoch, model, val_loader, criterion, val_losses, val_accuracies, device)

        # update learning rate schudule
        scheduler.step(epoch + 1)
    
        # print stats
        print(f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.2f}% | "
            f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accuracies[-1]:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # save model
    torch.save(model.state_dict(), "gfnet_alzheimers.pth")  # Save trained model weights
    print("Model saved as gfnet_alzheimers.pth")

    # plot the training stats
    plot_stats(args.epochs, train_losses, val_losses, train_accuracies, val_accuracies) 


if __name__ == "__main__":
    main()




