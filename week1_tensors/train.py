"""
Hafta 1: Train/Val Split + L2 Regularization
Gerçekçi train pipeline: early stopping, validation tracking.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from common.utils import get_device, print_section, set_seed


class LinearRegression(nn.Module):
    """Linear regression model."""

    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def generate_data(n_samples: int = 500, noise: float = 0.15):
    """Generate larger dataset for train/val split."""
    set_seed(42)
    device = get_device()

    X = torch.rand(n_samples, 1, device=device)
    true_w, true_b = 3.0, 2.0
    y = true_w * X + true_b + noise * torch.randn_like(X)

    return X, y


def create_dataloaders(X, y, train_ratio: float = 0.8, batch_size: int = 32):
    """
    Split data into train/val and create DataLoaders.

    Args:
        X, y: Full dataset
        train_ratio: Ratio of training data
        batch_size: Batch size

    Returns:
        train_loader, val_loader
    """
    # 🚧 TODO: TensorDataset oluştur
    dataset = TensorDataset(X, y)

    # 🚧 TODO: Train/val split (random_split kullan)
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    # 🚧 TODO: DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


def train_with_validation(
    model,
    train_loader,
    val_loader,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    epochs: int = 200,
    patience: int = 10,
):
    """
    Train with validation and early stopping.

    Args:
        model: nn.Module
        train_loader, val_loader: DataLoaders
        lr: Learning rate
        weight_decay: L2 regularization strength
        epochs: Max epochs
        patience: Early stopping patience

    Returns:
        train_losses, val_losses: Loss histories
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()

    # 🚧 TODO: Optimizer (weight_decay = L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print_section(f"Training with L2 Reg (weight_decay={weight_decay})")
    print(f"Max epochs: {epochs}, Early stopping patience: {patience}\n")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | Patience: {patience_counter}/{patience}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
            break

    return train_losses, val_losses


def plot_train_val_curves(train_losses, val_losses):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2, alpha=0.8)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Train vs Validation Loss (with L2 Reg & Early Stopping)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("week1_tensors/train_val_curve.png", dpi=120)
    print("\n💾 Plot saved: train_val_curve.png")
    plt.show()


def main():
    # Data
    X, y = generate_data(n_samples=500, noise=0.15)
    train_loader, val_loader = create_dataloaders(X, y, train_ratio=0.8, batch_size=32)

    print(f"📊 Train samples: {len(train_loader.dataset)}")
    print(f"📊 Val samples: {len(val_loader.dataset)}\n")

    # Model
    device = get_device()
    model = LinearRegression(input_dim=1).to(device)

    # Train
    train_losses, val_losses = train_with_validation(
        model,
        train_loader,
        val_loader,
        lr=0.1,
        weight_decay=0.01,  # L2 regularization
        epochs=200,
        patience=15,
    )

    # Results
    print_section("Final Results")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Learned w: {model.linear.weight.item():.4f} (true: 3.00)")
    print(f"Learned b: {model.linear.bias.item():.4f} (true: 2.00)\n")

    # Plot
    plot_train_val_curves(train_losses, val_losses)

    # ✅ Success criterion
    if val_losses[-1] < 0.5:
        print("✅ SUCCESS: Val MSE < 0.5 achieved!")
    else:
        print("⚠️  Val MSE >= 0.5, consider tuning hyperparameters.")


if __name__ == "__main__":
    main()
