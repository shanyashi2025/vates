import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

from vates.finmath import BlackScholesCalculator


def main():
    if not has_matplotlib:
        raise ImportError(f"need to install `matplotlib` library (pip install matplotlib)")
    import matplotlib.pyplot as plt

    base_dir = Path(__file__).resolve().parent

    spot_prices = np.linspace(80, 120, 100)
    times_to_expiry = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(spot_prices, times_to_expiry)
    _shape = X.shape
    Z = np.zeros(shape=_shape)

    K = 100
    r = 0.05
    sigma = 0.3

    for i in range(_shape[0]):
        for j in range(_shape[1]):
            s, tau = X[i, j], Y[i, j]
            Z[i, j] = BlackScholesCalculator.price(call_or_put="call", s=s, k=K, r=r, sigma=sigma, tau=tau)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='black', linewidth=0.1, alpha=0.9)

    ax.set_xlabel('Spot Price (S)', labelpad=10)
    ax.set_ylabel('Time to Expiry (T)', labelpad=10)
    ax.set_zlabel('Call Option Value', labelpad=10)
    ax.set_title(f'Call Option Value (Black-Scholes)\n(K={K:.2f}, r={r:.2%}, $\sigma$={sigma:.2%})', fontsize=13, pad=15)

    ax.set_xlim(80, 120)
    ax.set_ylim(0, 1)

    ax.view_init(elev=25, azim=-135)

    try:
        ax.xaxis._axinfo["grid"]['color'] = 'gray'
        ax.xaxis._axinfo["grid"]['linestyle'] = '--'
        ax.yaxis._axinfo["grid"]['color'] = 'gray'
        ax.yaxis._axinfo["grid"]['linestyle'] = '--'
        ax.zaxis._axinfo["grid"]['color'] = 'gray'
        ax.zaxis._axinfo["grid"]['linestyle'] = '--'
    except Exception:
        ax.grid(True, linestyle='--', color='gray', alpha=0.5)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()
    plt.savefig(base_dir / "Black-Scholes Option Pricing Model.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
