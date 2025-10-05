"""
Hafta 1: Linear Regression (nn.Module ile)
PyTorch'un yüksek seviye API'sini kullanarak daha temiz kod.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common.utils import get_device, print_section, set_seed


class LinearRegression(nn.Module):
    """Simple linear regression model: y = wx + b"""

    def __init__(self, input_dim: int = 1):
        super().__init__()
        # 🚧 TODO: nn.Linear layer tanımla
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 🚧 TODO: Forward pass
        return self.linear(x)


def generate_data(n_samples: int = 100, noise: float = 0.1):
    """Generate synthetic linear data."""
    set_seed(42)
    device = get_device()

    X = torch.rand(n_samples, 1, device=device)
    true_w, true_b = 3.0, 2.0
    y = true_w * X + true_b + noise * torch.randn_like(X)

    return X, y, (true_w, true_b)


def train_model(model, X, y, lr: float = 0.01, epochs: int = 1000):
    """
    Train model using PyTorch optimizer.

    Args:
        model: nn.Module
        X, y: Data
        lr: Learning rate
        epochs: Training epochs

    Returns:
        losses: List of loss values
    """
    # 🚧 TODO: Loss function tanımla (MSE)
    criterion = nn.MSELoss()

    # 🚧 TODO: Optimizer tanımla (Adam veya SGD)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    print_section("Training with nn.Module & Adam")
    print(f"Learning rate: {lr}, Epochs: {epochs}\n")

    for epoch in range(epochs):
        # 🚧 TODO: Forward pass
        y_pred = model(X)

        # 🚧 TODO: Loss hesapla
        loss = criterion(y_pred, y)

        # 🚧 TODO: Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            w = model.linear.weight.item()
            b = model.linear.bias.item()
            print(f"Epoch {epoch + 1:4d} | Loss: {loss.item():.6f} | w: {w:.4f}, b: {b:.4f}")

    return losses


def main():
    # Data
    X, y, true_params = generate_data(n_samples=100, noise=0.2)
    device = X.device

    # Model
    model = LinearRegression(input_dim=1).to(device)
    print(f"📦 Model: {model}")
    print(f"🔢 Parameters: {sum(p.numel() for p in model.parameters())}\n")

    # Train
    losses = train_model(model, X, y, lr=0.1, epochs=1000)

    # Results
    print_section("Final Results")
    w_learned = model.linear.weight.item()
    b_learned = model.linear.bias.item()
    print(f"Learned w: {w_learned:.4f} (true: {true_params[0]:.2f})")
    print(f"Learned b: {b_learned:.4f} (true: {true_params[1]:.2f})")
    print(f"Final MSE: {losses[-1]:.6f}\n")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss (nn.Module)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("week1_tensors/linreg_module_loss.png", dpi=120)
    print("💾 Plot saved: linreg_module_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
