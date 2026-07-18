import json
import math
import numpy as np
import pandas as pd
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

import vates
from vates import StochExecutor, ProjModelEngine, TDimVariable


def port_monte_carlo_proj(model: ProjModelEngine, risk_free_rate, n_assets, mu, sigma, corr_matrix, portfolio_params):
    m_rf_rate = (1 + risk_free_rate) ** (1/12) - 1

    t, p = model.time, model.period

    if t == 0:
        model.portfolios = {}

        for key, params in portfolio_params.items():
            ibal: float = params['init_balance']
            model.portfolios[key] = {
                'weight': params['weight'],
                'rfawgt': params['rfawgt'],
                'rebalance_freq': params['rebalance_freq'],
                'asset_amount': ibal * params['weight'],
                'rfa_amount': ibal * params['rfawgt'],
                'balance': TDimVariable("balance", model_engine=model, owner=key, group='MonteCarlo'),
                'return': TDimVariable("return", model_engine=model, owner=key, group='MonteCarlo'),
                'sum_ret': 0.0,
                'sum_sqret': 0.0
            }
            model.portfolios[key]['balance'][t] = ibal

        seed = model.SIMULATION
        model.rng = np.random.default_rng(seed)

    else:  # t > 0

        dt = 1 / 12

        if n_assets > 1:
            z = vates.finmath.multivariate_standard_normal(corr_matrix, rng=model.rng, bypass_valid_corr=True)
        else:
            z = model.rng.standard_normal(1)  # note: standard_normal(1) >> np.ndarray, standard_normal(0) >> float

        r = np.array([vates.finmath.geometric_brownian_motion(mu=mu[i], sigma=sigma[i], dt=dt, z=z[i])
                      for i in range(n_assets)]) - 1  # # S(t)/S(t-1) - 1

        for _, port_var in model.portfolios.items():
            port_var['asset_amount'] += port_var['asset_amount'] * r
            port_var['rfa_amount'] += port_var['rfa_amount'] * m_rf_rate
            port_var['balance'][p] = sum(port_var['asset_amount']) + port_var['rfa_amount']
            ret = port_var['balance'][p] / port_var['balance'][p - 1] - 1
            port_var['return'][p] = ret
            port_var['sum_ret'] += ret
            port_var['sum_sqret'] += ret ** 2

            if p.month % (12 // port_var['rebalance_freq']) == 0:  # reblance at this month
                port_var['asset_amount'] = port_var['balance'][p] * port_var['weight']
                port_var['rfa_amount'] = port_var['balance'][p] * port_var['rfawgt']

    if t == model.MAX_T:
        for port_name, port_var in model.portfolios.items():
            twrr = port_var['balance'][t] ** (12 / t) - 1  # time-weighted rate of return (TWRR)
            mean_return = port_var['sum_ret'] / t
            ms_return = port_var['sum_sqret'] / t  # mean squared return
            std_return = math.sqrt(ms_return - mean_return ** 2)  # variance = E(X^2) - (E(X))^2
            mean_return = (1 + mean_return) ** 12 - 1  # annualized
            std_return = std_return * math.sqrt(12)  # annualized
            sharp_ratio = (mean_return - risk_free_rate) / std_return

            port_var['twrr'] = vates.ConstVariable('twrr', model_engine=model, owner=port_name, group='MonteCarlo')
            port_var['mean_return'] = vates.ConstVariable('mean_return', model_engine=model, owner=port_name, group='MonteCarlo')
            port_var['std_return'] = vates.ConstVariable('std_return', model_engine=model, owner=port_name, group='MonteCarlo')
            port_var['sharp_ratio'] = vates.ConstVariable('sharp_ratio', model_engine=model, owner=port_name, group='MonteCarlo')

            port_var['twrr'][...] = twrr
            port_var['mean_return'][...] = mean_return
            port_var['std_return'][...] = std_return
            port_var['sharp_ratio'][...] = sharp_ratio


def port_monte_carlo_stoch(simulations: str, start_year: int, start_month: int, end_year: int, scenario: str,
                           workspace_directory: str, input_directories: list[str], results_directory: str | None = None,
                           max_workers: int | None = None,
                           model_name: str = "monte_carlo", description: str = "Portfolio Monte Carlo simulation"):
    model = StochExecutor(
        model_name=model_name,
        description=f"{description}, simulations: {simulations}, scenario: '{scenario}', "
                    f"from {start_year}/{start_month} to {end_year}/12."
    )
    model.bind_projection(port_monte_carlo_proj)
    model.configure_run(
        simulations=simulations,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=12,
        scenario=scenario,
        workspace_directory=workspace_directory,
        input_directories=input_directories,
        results_directory=results_directory,
        max_workers=max_workers,
    )

    negative_weight_allowed = False
    negatvie_rfawgt_allowed = False
    # parameters
    df = model.read_csv('parameters.csv', index_col="parameter")
    risk_free_rate = df.at['risk_free_rate', 'value']
    # assets
    df = model.read_csv('assets.csv', index_col="asset_name")
    asset_name_list = df.index.tolist()
    n_assets = len(asset_name_list)
    mu, sigma, corr_matrix = np.zeros(n_assets), np.zeros(n_assets), np.zeros(shape=(n_assets, n_assets))
    for i in range(n_assets):
        asset_name = asset_name_list[i]
        mu[i] = math.log(1 + df.at[asset_name, 'expected_return'])  # convert to continually compounded
        sigma[i] = df.at[asset_name, 'standard_deviation']
        for j in range(n_assets):
            corr_matrix[i, j] = df.at[asset_name, asset_name_list[j]]
    valid, msg = vates.finmath.validate_corr_matrix(corr_matrix)
    if not valid: raise ValueError(msg)

    # portfolios
    df = model.read_csv('portfolios.csv', index_col="portfolio_name")
    portfolio_params: dict[str, dict] = {}

    for port in df.index:
        init_balance = df.at[port, 'initial_balance']
        rebalance_freq = df.at[port, 'rebalance_freq']
        weight = np.array([df.at[port, asset_name_list[i]] for i in range(n_assets)])
        rfawgt = 1 - sum(weight)
        if not negative_weight_allowed and any(weight < 0):
            raise ValueError(f'Negative asset weight is NOT allowed. Please revise input of portfolio {port}.')
        if not negatvie_rfawgt_allowed and rfawgt < 0:
            raise ValueError(f'Sum of asset weight exceeds 100%. Please revise input of portfolio {port}.')
        portfolio_params[port] = {
            'init_balance': init_balance,
            'rebalance_freq': rebalance_freq,
            'weight': weight,
            'rfawgt': rfawgt,
        }

    model.run(
        projection_args={
            "risk_free_rate": risk_free_rate,
            "n_assets": n_assets,
            "mu": mu,
            "sigma": sigma,
            "corr_matrix": corr_matrix,
            "portfolio_params": portfolio_params,
        }
    )

    # print(model.proj_result())

    if not has_matplotlib:
        print(f"`matplotlib` library is not installed, can't plot simulation path.")
    else: # has_matplotlib:
        # === Plot portfolio balance simualtion path ===
        import glob
        import matplotlib.pyplot as plt

        # === Plot portfolio balance simualtion path ===
        print(f"Started to plot simulation path ...")
        stoch_files = glob.glob(f'{model.results_directory_path}/{model.MODEL_NAME}*.stoch.csv')
        df_all = pd.concat((pd.read_csv(f) for f in stoch_files), ignore_index=True)

        cols = [] if model.START_MONTH == 12 else [str(model.START_YEAR * 100 + model.START_MONTH)]
        cols.extend(list(str(y) for y in range(model.START_YEAR, model.END_YEAR + 1)))

        n = len(portfolio_params)
        fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(18, 5 * n))
        axes = axes.flatten()

        for i, port in enumerate(portfolio_params):
            rows = ((df_all["group"] == 'MonteCarlo') & (df_all["owner"] == port) & (df_all["variable"] == 'balance'))
            df = df_all.loc[rows, cols]

            mean = df.mean()
            p5, p95 = df.quantile(0.05), df.quantile(0.95)
            p10, p90 = df.quantile(0.1), df.quantile(0.9)
            p25, p75 = df.quantile(0.25), df.quantile(0.75)
            p50 = df.quantile(0.5)

            ax1, ax2, ax3 = axes[i * 3], axes[i * 3 + 1], axes[i * 3 + 2]
            # --- (1) All simulation paths ---
            for s in range(len(df)):
                ax1.plot(cols, df.iloc[s], alpha=0.3)
            ax1.set_title(f"{port}\nAll Simulation Paths")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Portfolio Balance")

            # --- (2) Mean + confidence band ---
            ax2.plot(cols, mean, label="mean", linestyle='solid')
            ax2.plot(cols, p10, label="10th", linestyle='dotted')
            ax2.plot(cols, p25, label="25th", linestyle='dashdot')
            ax2.plot(cols, p50, label="50th", linestyle='dashed')
            ax2.plot(cols, p75, label="75th", linestyle='dashdot')
            ax2.plot(cols, p90, label="90th", linestyle='dotted')

            ax2.fill_between(cols, p5, p95, alpha=0.2, label="5–95%")
            ax2.set_title(f"{port}\nMean & Confidence Interval")
            ax2.set_xlabel("Year")
            ax2.legend()
            ax2.sharey(ax1)

            # --- (3) End balance histogram ---
            data = df[str(model.END_YEAR)].values
            p_low, p_high = np.percentile(data, [2.5, 97.5])  # 95% interval

            ax3.hist(data, density=True, bins=20)
            ax3.axvline(p_low, color='r', linestyle='--', label=f'2.5th percentile = {p_low:,.0f}')
            ax3.axvline(p_high, color='g', linestyle='--', label=f'97.5th percentile = {p_high:,.0f}')
            ax3.set_title(f"{port}\nPortfolio End Balance Distribution (95% Interval)")
            ax3.set_xlabel("End Balance")
            ax3.set_ylabel("Frequency")
            ax3.legend()

        plt.tight_layout()
        output_png = f'{model.results_directory_path}/{model.MODEL_NAME}_simulation_path.png'
        plt.savefig(output_png, dpi=300)
        plt.close()
        print(f"simulation path figure saved to : '{output_png}'")

def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    port_monte_carlo_stoch(**kwargs)


if __name__ == "__main__":
    main()

