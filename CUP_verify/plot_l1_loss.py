
import numpy as np
import matplotlib.pyplot as plt

def loss(x, k):
    return np.log1p(np.exp(k*(0.05 - x))) + np.log1p(np.exp(k*(x - 0.2)))

x = np.linspace(-0.1, 0.35, 500)
k_values = [10, 100, 1000]

plt.figure(figsize=(8, 12))  # Taller figure for 3 stacked plots

for i, k in enumerate(k_values, 1):
    y = loss(x, k)
    plt.subplot(3, 1, i)
    plt.plot(x, y, label=f'Loss Function k={k}')
    plt.axvline(0.05, color='gray', linestyle='--', label='x = 0.05')
    plt.axvline(0.2, color='gray', linestyle='--', label='x = 0.2')
    plt.xlabel('x')
    plt.ylabel('L(x)')
    plt.title(f'Smooth Differentiable Loss Function (k={k})')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
