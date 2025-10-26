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

def visualise_results(model, device, num_samples=9, save_path="alzheimers_predictions.png"):
    """
    Display and save random Alzheimer classification predictions.
    Includes predicted probability and uses a distinct grid layout.
    """
    from dataset import BrainMRIDataset, DATASET_PATH, TEST_TRANSFORM

    model.eval()
    dataset = BrainMRIDataset(DATASET_PATH, train=False, transform=TEST_TRANSFORM)
    label_map = {0: "NC", 1: "AD"}

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, true_label = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            pred_label = label_map[int(pred.item())]
            confidence = conf.item()

        # Plot grayscale image
        img_np = image.squeeze(0).cpu().numpy()
        axes[i].imshow(img_np, cmap="bone")
        axes[i].set_title(
            f"Pred: {pred_label} ({confidence*100:.1f}%)\nTrue: {label_map[true_label]}",
            fontsize=9,
            color="green" if pred_label == label_map[true_label] else "red"
        )
        axes[i].axis("off")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Alzheimer’s MRI Predictions (GFNetAlzheimers)", fontsize=12, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(save_path, dpi=150)
    print(f"Saved prediction grid to: {save_path}")


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

    # visualise predictions
    visualise_results(model, device, num_samples=9)

if __name__ == "__main__":
    main()