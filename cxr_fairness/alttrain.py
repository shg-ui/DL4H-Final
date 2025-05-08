import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import densenet121
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

def create_densenet121_model(pretrained=True):
    """
    Creates a DenseNet121 model pre-trained on ImageNet, and replaces
    the final classification layer with a custom 2-layer fully connected
    network.

    Args:
        pretrained (bool, optional): If True, loads pre-trained weights for
            the DenseNet121 layers. Defaults to True.

    Returns:
        torch.nn.Module: The modified DenseNet121 model.
    """
    # Load the DenseNet121 model
    densenet121 = models.densenet121(pretrained=pretrained)

    # Get the number of input features of the original last layer
    in_features = densenet121.classifier.in_features

    # Create a new classifier (2-layer fully connected network)
    new_classifier = nn.Sequential(
        nn.Linear(in_features, 384),  # First fully connected layer
        nn.ReLU(),
        nn.Linear(384, 1),          # Second fully connected layer, 1 output
        nn.Sigmoid()                # Apply Sigmoid for binary classification (0 to 1)
    )

    # Replace the model's classifier with our new classifier
    densenet121.classifier = new_classifier

    return densenet121



def train_model(model, train_loader, optimizer, loss_fn, epochs, device):
    """
    Trains the given model using the provided data loader, optimizer, and loss function.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_fn (torch.nn.Module): The loss function to optimize.
        epochs (int): The number of epochs to train for.
        device (torch.device): The device to train on (CPU or GPU).

    Returns:
        list: A list containing the training losses for each epoch.
    """
    model.train()  # Set the model to training mode
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            # Ensure labels are float and have the correct shape for binary cross-entropy
            labels = labels.float().view(-1, 1)
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}")

    return train_losses


def evaluate_model(model, data_loader, device):
    """
    Evaluates the trained model on the given data loader using the Area Under the ROC Curve (AUC-ROC).

    Args:
        model (torch.nn.Module): The trained neural network model.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the evaluation data.
        device (torch.device): The device to evaluate on (CPU or GPU).

    Returns:
        float: The AUC-ROC score.
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_probs = []

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float().view(-1, 1) # Ensure correct shape for comparison
            outputs = model(images)
            probabilities = outputs.cpu().numpy()  # Move to CPU for sklearn
            labels = labels.cpu().numpy()
            all_probs.append(probabilities)
            all_labels.append(labels)

    # Concatenate all predictions and labels
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Calculate AUC-ROC
    auc_roc = roc_auc_score(all_labels, all_probs)
    return auc_roc



def main():
    """
    Main function to demonstrate the training and evaluation loop with the DenseNet121 model.
    """
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    image_size = 224

    # Create the DenseNet121 model
    model = create_densenet121_model(pretrained=True).to(device)

    # Example data (replace with your actual chest X-ray data)
    num_samples = 100
    dummy_images = torch.randn(num_samples, 3, image_size, image_size)
    dummy_labels = torch.randint(0, 2, (num_samples, 1)).float()

    # Split data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        dummy_images, dummy_labels, test_size=0.2, random_state=42
    )

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses = train_model(model, train_loader, optimizer, loss_fn, epochs, device)

    # Evaluate the model
    auc_roc = evaluate_model(model, val_loader, device)
    print(f"Validation AUC-ROC: {auc_roc:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'chest_xray_model.pth')
    print("Trained model saved to chest_xray_model.pth")


if __name__ == "__main__":
    main()
