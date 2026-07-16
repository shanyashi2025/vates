import csv
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

from vates.utils import search_efficient_frontier

def _min_std_meeting_ret_tar(wgts: np.ndarray, rets: np.ndarray, stds: np.ndarray, ret_tar: float) -> dict | None:
    """minimum standard deviation meeting return target"""
    mask = rets >= ret_tar
    if mask.any():
        idx = np.argmin(stds[mask])
        return {'wgt': wgts[mask][idx], 'ret': rets[mask][idx], 'std': stds[mask][idx]}
    else:
        return None

def _max_ret_within_std_tol(wgts: np.ndarray, rets: np.ndarray, stds: np.ndarray, std_tol: float) -> dict | None:
    """maximum return within standard deviation tolerance"""
    mask = stds <= std_tol
    if mask.any():
        idx = np.argmax(stds[mask])
        return {'wgt': wgts[mask][idx], 'ret': rets[mask][idx], 'std': stds[mask][idx]}
    else:
        return None


def main():
    if not has_matplotlib:
        raise ImportError(f"need to install `matplotlib` library (pip install matplotlib)")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent

    # Efficient frontier and portfolio optimizer
    df = pd.read_csv(base_dir / 'parameters.csv', index_col='parameter')
    risk_free_rate = df.at['risk_free_rate', 'value']
    ret_tar = df.at['return_target', 'value']
    std_tol = df.at['standard_deviation_tolerance', 'value']
    df = pd.read_csv(base_dir / 'assets.csv', index_col="asset_name")
    asset_name_list = df.index.tolist()
    n_assets = len(asset_name_list)
    mu, sigma, corr_matrix = np.zeros(n_assets), np.zeros(n_assets), np.zeros(shape=(n_assets, n_assets))
    for i in range(n_assets):
        asset_name = asset_name_list[i]
        mu[i] = df.at[asset_name, 'expected_return']
        sigma[i] = df.at[asset_name, 'standard_deviation']
        for j in range(n_assets):
            corr_matrix[i, j] = df.at[asset_name, asset_name_list[j]]

    max_points = int((mu.max() - mu.min()) / 0.001)
    wgts, rets, stds = search_efficient_frontier(mu, sigma, corr_matrix, n_points=min(max_points, 200))

    sharpe_ratios = (rets - risk_free_rate) / stds
    tan_idx = np.argmax(sharpe_ratios)  # index of tangency portfolio
    gmv_idx = np.argmin(stds)  # index of global minimum variance portfolio
    mins_rtar = _min_std_meeting_ret_tar(wgts, rets, stds, ret_tar)  # minimum std meeting return target
    maxr_stol = _max_ret_within_std_tol(wgts, rets, stds, std_tol)  # maximum return within std tolerance

    # === Output ===
    output_file_path = base_dir / "efficient_frontier_portfolios.csv"
    with open(output_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        header = ['#', 'Portfolio'] + asset_name_list + ['Expected Return', 'Standard Deviation', 'Sharpe Ratio']
        writer.writerow(header)
        # --- (1) risk-free asset ---
        content = [0, 'risk-free asset'] + [0] * n_assets + [risk_free_rate, 0, 0]
        writer.writerow(content)
        # --- (2) individual assets ---
        for i in range(n_assets):
            content = [i + 1, asset_name_list[i]] + \
                      [1 * (1 if j == i else 0) for j in range(n_assets)] + \
                      [mu[i], sigma[i], (mu[i] - risk_free_rate) / sigma[i]]
            writer.writerow(content)
        # --- (3) tangency portfolio ---
        content = [n_assets + 1, 'tangency portfolio'] + \
                  wgts[tan_idx].round(8).tolist() + \
                  [rets[tan_idx], stds[tan_idx], sharpe_ratios[tan_idx]]
        writer.writerow(content)
        # --- (4) global minimum variance portfolio ---
        content = [n_assets + 2, 'global minimum variance portfolio (GMV)'] + \
                  wgts[gmv_idx].round(8).tolist() + \
                  [rets[gmv_idx], stds[gmv_idx], sharpe_ratios[gmv_idx]]
        writer.writerow(content)
        # --- (5) minimum standard deviation meeting return target ---
        if mins_rtar:
            ret, std = mins_rtar['ret'], mins_rtar['std']
            content = [n_assets + 3, 'min standard deviation meeting return target'] + \
                      mins_rtar['wgt'].round(8).tolist() + \
                      [ret, std, (ret - risk_free_rate) / std]
        else:
            content = [n_assets + 3, '(None) min standard deviation meeting return target']
        writer.writerow(content)
        # --- (6) maximum return within standard deviation tolerance ---
        if maxr_stol:
            ret, std = maxr_stol['ret'], maxr_stol['std']
            content = [n_assets + 4, 'max return within standard deviation tolerance'] + \
                      maxr_stol['wgt'].round(8).tolist() + \
                      [ret, std, (ret - risk_free_rate) / std]
        else:
            content = [n_assets + 4, '(None) max return within standard deviation tolerance']
        writer.writerow(content)
        # --- (7) efficient portfolios ---
        for i in range(len(wgts)):
            content = [n_assets + 5 + i, f'portforlio #{i + 1}'] + \
                      wgts[i].round(8).tolist() + \
                      [rets[i], stds[i], sharpe_ratios[i]]
            writer.writerow(content)

    # === Plot ===
    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(stds, rets, color='b', s=5, alpha=0.6)
    plt.axhline(y=ret_tar, linestyle='--', linewidth=0.5, label=f'{ret_tar * 100:.2f}% (return target)')
    plt.axvline(x=std_tol, linestyle='--', linewidth=0.5, label=f'{std_tol * 100:.2f}% (standard deviation tolerance)')
    # --- (1) individual assets ---
    for i in range(n_assets):
        plt.scatter(sigma[i], mu[i], color='red', s=10, zorder=3)
        plt.annotate(asset_name_list[i], (float(sigma[i]), float(mu[i])),
                     xytext=(5, 3), textcoords='offset points', fontsize=6, ha='left', va='bottom')
    # --- (2) risk-free asset ---
    plt.scatter(0, risk_free_rate, color="green", s=5)
    plt.annotate(f'Risk-free ({risk_free_rate * 100:.2f}%)', (0, risk_free_rate),
                 xytext=(5, 3), textcoords='offset points', fontsize=6, ha='left', va='bottom')
    # --- (3) tangency portfolio ---
    plt.scatter(stds[tan_idx], rets[tan_idx], color='orange', s=10, zorder=3)
    plt.annotate('Tangency', (float(stds[tan_idx]), float(rets[tan_idx])),
                 xytext=(5, 3), textcoords='offset points', fontsize=6, ha='left', va='bottom')
    # --- (4) capital allocation line (CAL) ---
    plt.plot(
        [0, stds[tan_idx]],
        [risk_free_rate, rets[tan_idx]],
        color='black', linewidth=0.8, linestyle='-',
        zorder=2, label="Capital Allocation Line"
    )

    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel("Standard Deviation", fontsize=7)
    plt.ylabel("Expected Return", fontsize=7)
    plt.title("Efficient Frontier, Mean Variance Optimization (MVO) and Capital Allocation Line (CAL)", fontsize=8)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(alpha=0.25, linestyle="--")
    plt.margins(x=0.05, y=0.05)
    plt.tight_layout()

    output_file_path = base_dir / "efficient_frontier_plot.png"
    plt.savefig(output_file_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
