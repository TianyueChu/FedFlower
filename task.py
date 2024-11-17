import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import os
import json
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import numpy as np


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train_fn(model, train_loader, num_epochs=10, learning_rate=0.001, device="cuda"):
    """
    Train the CelebAMobileNet model and evaluate using precision, recall, and F1.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        nn.Module: Trained model.
    """
    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["image"].to(device)
            labels = batch["Demographic_Label"].to(device)  # Integer labels

            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save labels and predictions for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Print batch details
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%"
                )

        # Get unique labels to dynamically determine target names
        unique_labels = sorted(list(set(all_labels)))
        target_names = [f"Class {label}" for label in unique_labels]

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Compute additional metrics
        precision = precision_score(all_labels, all_predictions, average="macro",zero_division=0)
        recall = recall_score(all_labels, all_predictions, average="macro",zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average="macro",zero_division=0)

        print(
            f"Epoch {epoch + 1} Summary: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        # Print detailed classification report
        print("\nClassification Report:\n")
        print(classification_report(all_labels, all_predictions, target_names=target_names,zero_division=0))
        # Return metrics as a dictionary

    metrics = {
        "loss": train_loss,
        "accuracy": train_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics



def test_fn(model, test_loader, device="cuda", results_dir="./results/"):
    """
    Test the trained model, save evaluation metrics, and return the model.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to run the model on ("cuda" or "cpu").
        results_dir (str): Directory to save evaluation results.

    Returns:
        Tuple[float, Dict[str, Union[float, str]]]: A tuple containing:
            - avg_loss (float): The average loss over the test set.
            - metrics (dict): Dictionary with evaluation metrics (e.g., accuracy, precision, recall, etc.).

    """
    # Move model to evaluation mode
    model = model.to(device)
    model.eval()

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    # Metrics tracking
    correct = 0
    total = 0
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs = batch["image"].to(device)
            labels = batch["Demographic_Label"].to(device)  # Integer labels

            outputs = model(inputs)  # Forward pass

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            # Track accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    # Get unique labels to dynamically determine target names
    unique_labels = sorted(list(set(all_labels)))
    target_names = [f"Class {label}" for label in unique_labels]

    # Compute average loss
    avg_loss = total_loss / len(test_loader)

    # Compute metrics
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)  # added zero_division to handle cases where a class is not predicted
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    classification_rep = classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0)  # using dynamic target_names and handling zero_division

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=unique_labels)
    conf_matrix_file = os.path.join(results_dir, "confusion_matrix.npy")
    np.save(conf_matrix_file, conf_matrix)

    # Save metrics to results folder
    metrics = {
        "Loss": avg_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Classification Report": classification_rep,
        "Confusion Matrix File": conf_matrix_file,
    }

    metrics_file = os.path.join(results_dir, "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Test Results:\nAccuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_rep)

    return avg_loss, metrics