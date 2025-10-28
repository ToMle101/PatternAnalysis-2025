import torch
from dataset import get_dataloader, BrainMRIDataset, DATASET_PATH, TEST_TRANSFORM
from modules import GFNetAlzheimers
import random
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import numpy as np

class Visualiser:
    """
    Provides visualisation tools for evaluating and interpreting
    the performance of the model. 
    
    Supports confusion matrix, ROC, PR curve, and prediction visualizations.
    """
    def __init__(self, save_prefix="alzheimers"):
        """
        initialise visualiser with a prefix for saving outputs

        Args:
            save_prefix (str): Prefix for saved output files.
        """
        self.save_prefix = save_prefix
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Plot and save a confusion matrix heatmap.

        Args:
            y_true (array-like): True class labels.
            y_pred (array-like): Predicted class labels.
            class_names (list): Names of the classes (e.g., ["NC", "AD"]).
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Annotate each cell with count value
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")

        plt.tight_layout()
        plt.savefig(f"{self.save_prefix}_confusion_matrix.png", dpi=150)
        print("Saved confusion matrix")
    
    def plot_roc(self, y_true, y_prob):
        """
        Plot and save the ROC (Receiver Operating Characteristic) curve.

        Args:
            y_true (array-like): True class labels (binary).
            y_prob (array-like): Predicted probabilities for the positive class.

        Returns:
            float: Computed Area Under Curve (AUC) value.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")  # Diagonal reference line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_prefix}_roc.png", dpi=150)
        print("Saved ROC curve")
        return roc_auc

    def plot_pr(self, y_true, y_prob):
        """
        Plot and save the Precision–Recall curve.

        Args:
            y_true (array-like): True class labels.
            y_prob (array-like): Predicted probabilities for the positive class.

        Returns:
            float: Average Precision (AP) score.
        """
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        plt.figure()
        plt.plot(rec, prec, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.save_prefix}_pr.png", dpi=150)
        print("Saved PR curve")
        return ap

    def plot_sample_predictions(self, model, device, dataset, num_samples=9):
        """
        Display and save a grid of random sample predictions.

        Args:
            model (torch.nn.Module): Trained PyTorch model.
            device (torch.device): Computation device (CPU or GPU).
            dataset (Dataset): Dataset object containing MRI images and labels.
            num_samples (int): Number of samples to visualize.
        """
        label_map = {0: "NC", 1: "AD"}
        model.eval()

        # Prepare a 3x3 grid of sample predictions
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        axes = axes.flatten()

        for i in range(num_samples):
            # Randomly pick an image from dataset
            idx = random.randint(0, len(dataset) - 1)
            image, true_label = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            # Forward pass to get prediction
            with torch.no_grad():
                output = model(image_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                pred_label = label_map[int(pred.item())]
                confidence = conf.item()

            # Convert image tensor to numpy for plotting
            img_np = image.squeeze(0).cpu().numpy()
            axes[i].imshow(img_np, cmap="bone")

            # Annotate with prediction info (green if correct, red if wrong)
            axes[i].set_title(
                f"Pred: {pred_label} ({confidence*100:.1f}%)\nTrue: {label_map[true_label]}",
                fontsize=9,
                color="green" if pred_label == label_map[true_label] else "red"
            )
            axes[i].axis("off")

        # Hide extra subplots if fewer than 9 samples
        for j in range(num_samples, len(axes)):
            axes[j].axis("off")

        fig.suptitle("Alzheimer’s MRI Predictions (GFNetAlzheimers)", fontsize=12, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        save_path = f"{self.save_prefix}_sample_predictions.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved prediction grid: {save_path}")

    def plot_misclassified(self, model, device, dataset, n=12):
        """
        Display and save a grid of misclassified samples.

        Args:
            model (torch.nn.Module): Trained PyTorch model.
            device (torch.device): Computation device.
            dataset (Dataset): Dataset to evaluate.
            n (int): Number of misclassified samples to show.
        """
        model.eval()
        imgs, titles = [], []
        label_map = {0: "NC", 1: "AD"}

        # Collect misclassified samples
        with torch.no_grad():
            for i in range(len(dataset)):
                image, true_label = dataset[i]
                x = image.unsqueeze(0).to(device)
                probs = torch.softmax(model(x), dim=1)
                conf, pred = torch.max(probs, 1)
                pred_label = pred.item()

                # Save if model misclassified
                if pred_label != true_label:
                    imgs.append(image)
                    titles.append(
                        f"True:{label_map[true_label]} Pred:{label_map[pred_label]} ({conf.item()*100:.1f}%)"
                    )
                if len(imgs) >= n:
                    break

        if not imgs:
            print("No misclassifications found!")
            return

        # Plot grid of misclassified examples
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten()

        for i in range(len(imgs)):
            axes[i].imshow(imgs[i].squeeze(0), cmap="bone")
            axes[i].set_title(titles[i], fontsize=8, color="red")
            axes[i].axis("off")

        for j in range(len(imgs), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        save_path = f"{self.save_prefix}_misclassified.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved misclassified examples grid: {save_path}")

def evaluate(model, device, test_loader):
    """
    Evaluate model accuracy on the test dataset.

    Args:
        model (nn.Module): Trained GFNetAlzheimers model.
        device (torch.device): Device (CPU or GPU).
        test_loader (DataLoader): DataLoader for the test set.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - True labels
            - Predicted labels
            - Predicted probabilities (for positive class)
    """
    model.eval()
    correct, total = 0, 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

            # Store results for metric calculation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(classification_report(y_true, y_pred, target_names=["NC", "AD"]))
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

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

    # Get DataLoaders and load test dataset
    test_loader = get_dataloader(batch_size=8, train=False)
    test_dataset = BrainMRIDataset(DATASET_PATH, train=False, transform=TEST_TRANSFORM)

    # Evaluate the model
    y_true, y_pred, y_prob = evaluate(model, device, test_loader)

    # visualise predictions
    vis = Visualiser(save_prefix="alzheimers")
    vis.plot_confusion_matrix(y_true, y_pred, ["NC", "AD"])
    vis.plot_roc(y_true, y_prob)
    vis.plot_pr(y_true, y_prob)
    vis.plot_sample_predictions(model, device, test_dataset, num_samples=9)
    vis.plot_misclassified(model, device, test_dataset, n=12)

if __name__ == "__main__":
    main()