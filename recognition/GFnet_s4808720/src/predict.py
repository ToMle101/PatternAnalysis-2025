import torch
from dataset import get_dataloader, BrainMRIDataset, DATASET_PATH, TEST_TRANSFORM
from modules import GFNetAlzheimers
import random
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader

def evaluate(model, device, test_loader):
    """
    Evaluate model accuracy on the test dataset.

    Args:
        model (nn.Module): Trained GFNetAlzheimers model.
        device (torch.device): Device (CPU or GPU).
        test_loader (DataLoader): DataLoader for the test set.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    """
    Main training entry point.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model 
    model = GFNetAlzheimers(
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=2,
        embed_dim=128,
        depth=10,
        drop_rate=0.1,
        drop_path_rate=0.1,
        use_attention=True
    ).to(device)

    # load the saved weights
    model.load_state_dict(torch.load("gfnet_alzheimers.pth", map_location=device))
    model.eval()
    print("Model weights loaded successfully")

    # hyperparameters
    batch_size = 8 

    # Get DataLoaders
    test_loader = get_dataloader(batch_size=batch_size, train=False)

    # Evaluate the model
    evaluate(model, device, test_loader)


if __name__ == "__main__":
    main()