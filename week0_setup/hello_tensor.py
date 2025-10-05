"""
Hafta 0: PyTorch & MPS Device Doğrulama
Basit tensor işlemleri ile Metal Performance Shaders kontrolü.
"""

import sys

import torch

from common.utils import get_device, print_section


def main():
    print_section("PyTorch MPS (Metal) Doğrulama")

    # Device kontrolü
    device = get_device()

    # PyTorch versiyonu
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"🖥️  Python version: {sys.version.split()[0]}\n")

    # Basit tensor oluştur
    print("🔹 Creating random tensor...")
    x = torch.randn(3, 4, device=device)
    print(f"   Device: {x.device}")
    print(f"   Shape: {x.shape}")
    print(f"   Data:\n{x}\n")

    # Temel işlemler
    print("🔹 Matrix operations...")
    y = torch.randn(4, 4, device=device)
    z = torch.mm(x, y)  # Matrix multiplication
    print(f"   Result shape: {z.shape}")
    print(f"   Result:\n{z}\n")

    # Autograd denemesi
    print("🔹 Autograd check...")
    a = torch.tensor([2.0, 3.0], device=device, requires_grad=True)
    b = a * a * 3  # b = 3 * a^2
    loss = b.sum()
    loss.backward()
    print(f"   Input: {a}")
    print(f"   Gradient (db/da): {a.grad}")  # Should be 6*a = [12, 18]
    print()

    # Performans micro-benchmark
    print("🔹 Performance check (1000 matrix mults)...")
    import time

    x_bench = torch.randn(256, 256, device=device)
    y_bench = torch.randn(256, 256, device=device)

    # Warmup
    for _ in range(10):
        _ = torch.mm(x_bench, y_bench)

    start = time.perf_counter()
    for _ in range(1000):
        _ = torch.mm(x_bench, y_bench)
    if device.type == "mps":
        torch.mps.synchronize()  # MPS için senkronizasyon
    elif device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"   Time: {(end - start) * 1000:.2f}ms")
    print(f"   Avg per operation: {(end - start):.4f}ms\n")

    print("✅ All checks passed! PyTorch is ready for NovaDev.\n")


if __name__ == "__main__":
    main()
