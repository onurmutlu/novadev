"""
Hafta 1: Manuel Linear Regression (ilkel API)
y = wx + b formülünü sıfırdan yaz, manuel gradient descent.
"""

import matplotlib.pyplot as plt
import torch

from common.utils import get_device, print_section, set_seed


def generate_linear_data(n_samples: int = 100, noise: float = 0.1):
    """
    Sentetik linear data üret: y = 3x + 2 + noise

    Args:
        n_samples: Örnek sayısı
        noise: Gürültü miktarı

    Returns:
        X: (n_samples, 1) tensor
        y: (n_samples, 1) tensor
    """
    set_seed(42)
    device = get_device()

    # 🚧 TODO: X'i [0, 1] aralığında rastgele üret
    X = torch.rand(n_samples, 1, device=device)

    # 🚧 TODO: y = 3*X + 2 + noise (normal dağılımlı gürültü)
    true_w, true_b = 3.0, 2.0
    y = true_w * X + true_b + noise * torch.randn_like(X)

    return X, y, (true_w, true_b)


def train_manual(X, y, lr: float = 0.01, epochs: int = 1000):
    """
    Manuel gradient descent ile linear regression.

    Args:
        X: Input features
        y: Target values
        lr: Learning rate
        epochs: Training epochs

    Returns:
        w, b: Learned parameters
        losses: List of loss values
    """
    device = X.device

    # 🚧 TODO: Parametreleri başlat (requires_grad=True)
    w = torch.randn(1, 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)

    losses = []

    print_section("Manuel Gradient Descent")
    print(f"Learning rate: {lr}, Epochs: {epochs}\n")

    for epoch in range(epochs):
        # 🚧 TODO: Forward pass - y_pred hesapla
        y_pred = X @ w + b  # Matrix multiplication + bias

        # 🚧 TODO: Loss hesapla (MSE)
        loss = ((y_pred - y) ** 2).mean()

        # 🚧 TODO: Backward pass
        loss.backward()

        # 🚧 TODO: Manuel gradient descent (optimizer yok)
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

            # Gradientleri sıfırla
            w.grad.zero_()
            b.grad.zero_()

        losses.append(loss.item())

        # Her 100 epoch'ta logla
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:4d} | Loss: {loss.item():.6f} | w: {w.item():.4f}, b: {b.item():.4f}")

    return w.detach(), b.detach(), losses


def plot_results(X, y, w, b, losses, true_params):
    """Plot regression line and loss curve."""
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    w_val = w.cpu().item()
    b_val = b.cpu().item()
    true_w, true_b = true_params

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Regression line
    ax1.scatter(X_np, y_np, alpha=0.5, label="Data")
    x_line = torch.linspace(0, 1, 100)
    y_pred = w_val * x_line + b_val
    y_true = true_w * x_line + true_b
    ax1.plot(x_line, y_pred, "r-", linewidth=2, label=f"Learned: y={w_val:.2f}x+{b_val:.2f}")
    ax1.plot(x_line, y_true, "g--", linewidth=2, label=f"True: y={true_w:.2f}x+{true_b:.2f}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("y")
    ax1.set_title("Linear Regression Fit")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss curve
    ax2.plot(losses, linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Training Loss Curve")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("week1_tensors/linreg_manual_results.png", dpi=120)
    print("\n💾 Plot saved: linreg_manual_results.png")
    plt.show()


def main():
    # Data üret
    X, y, true_params = generate_linear_data(n_samples=100, noise=0.2)
    print(f"📊 Data shape: X={X.shape}, y={y.shape}")
    print(f"🎯 True parameters: w={true_params[0]:.2f}, b={true_params[1]:.2f}\n")

    # Train
    w, b, losses = train_manual(X, y, lr=0.1, epochs=1000)

    # Sonuçlar
    print_section("Final Results")
    print(f"Learned w: {w.item():.4f} (true: {true_params[0]:.2f})")
    print(f"Learned b: {b.item():.4f} (true: {true_params[1]:.2f})")
    print(f"Final MSE: {losses[-1]:.6f}\n")

    # Plot
    plot_results(X, y, w, b, losses, true_params)


if __name__ == "__main__":
    main()
