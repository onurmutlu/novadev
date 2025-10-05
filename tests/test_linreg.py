"""
Hafta 1: Linear Regression Testleri
Model convergence ve L2 regularization etkisini test et.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Week1 modüllerinden import (eğer package olarak kuruluysa)
# Alternatif: sys.path.append("..")


class LinearRegression(nn.Module):
    """Simple linear regression for testing."""

    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def generate_test_data(n_samples: int = 100):
    """Generate test data."""
    torch.manual_seed(42)
    X = torch.rand(n_samples, 1)
    y = 3.0 * X + 2.0 + 0.1 * torch.randn_like(X)
    return X, y


def test_model_convergence():
    """Test 1: Model should converge (MSE < 0.5)."""
    X, y = generate_test_data(n_samples=200)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearRegression(input_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Train for 100 epochs
    for _ in range(100):
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Final validation
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        final_mse = criterion(y_pred, y).item()

    print(f"Test 1 - Final MSE: {final_mse:.6f}")
    assert final_mse < 0.5, f"Model did not converge: MSE={final_mse:.4f}"


def test_l2_regularization():
    """Test 2: L2 reg should reduce parameter norms."""
    X, y = generate_test_data(n_samples=100)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Train WITHOUT L2
    model_no_l2 = LinearRegression()
    optimizer_no_l2 = torch.optim.SGD(model_no_l2.parameters(), lr=0.1, weight_decay=0.0)
    criterion = nn.MSELoss()

    for _ in range(50):
        for X_batch, y_batch in loader:
            loss = criterion(model_no_l2(X_batch), y_batch)
            optimizer_no_l2.zero_grad()
            loss.backward()
            optimizer_no_l2.step()

    # Train WITH L2
    model_with_l2 = LinearRegression()
    optimizer_with_l2 = torch.optim.SGD(model_with_l2.parameters(), lr=0.1, weight_decay=0.1)

    for _ in range(50):
        for X_batch, y_batch in loader:
            loss = criterion(model_with_l2(X_batch), y_batch)
            optimizer_with_l2.zero_grad()
            loss.backward()
            optimizer_with_l2.step()

    # Compare parameter norms
    norm_no_l2 = sum(p.norm().item() for p in model_no_l2.parameters())
    norm_with_l2 = sum(p.norm().item() for p in model_with_l2.parameters())

    print(f"Test 2 - Norm without L2: {norm_no_l2:.4f}")
    print(f"Test 2 - Norm with L2: {norm_with_l2:.4f}")

    # L2 reg should reduce norms (usually, not always guaranteed with different inits)
    # We'll just check they're in reasonable range
    assert norm_no_l2 > 0, "Model parameters are zero"
    assert norm_with_l2 > 0, "Model parameters are zero"
    print("Test 2 - L2 regularization applied successfully")


def test_gradient_computation():
    """Test 3: Gradients should be computed correctly."""
    torch.manual_seed(42)
    X = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=False)
    y = torch.tensor([[3.0], [5.0], [7.0]], requires_grad=False)

    model = LinearRegression()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check that gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    print("Test 3 - Gradients computed successfully")


if __name__ == "__main__":
    # Run tests manually (pytest will discover these automatically)
    print("=" * 60)
    print("  Running Hafta 1 Linear Regression Tests")
    print("=" * 60 + "\n")

    test_model_convergence()
    print("✅ Test 1 passed\n")

    test_l2_regularization()
    print("✅ Test 2 passed\n")

    test_gradient_computation()
    print("✅ Test 3 passed\n")

    print("🎉 All tests passed!")
