import json
import math
import numpy as np
import sys

import vates
from vates import StochExecutor, ProjModelEngine, TDepVariable


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
                'balance': TDepVariable("balance", model_engine=model, owner=key, group='MonteCarlo'),
                'return': TDepVariable("return", model_engine=model, owner=key, group='MonteCarlo'),
                'sum_ret': 0.0,
                'sum_sqret': 0.0
            }
            model.portfolios[key]['balance'][t] = ibal

        seed = model.SIMULATION
        model.rng = np.random.default_rng(seed)

    else:  # t > 0

        dt = 1 / 12

        if n_assets > 1:
            z = vates.utils.multivariate_standard_normal(corr_matrix, rng=model.rng, bypass_valid_corr=True)
        else:
            z = model.rng.standard_normal(1)  # note: standard_normal(1) >> np.ndarray, standard_normal(0) >> float

        r = np.array([vates.utils.geometric_brownian_motion(mu=mu[i], sigma=sigma[i], dt=dt, z=z[i])
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

            port_var['twrr'][t] = twrr
            port_var['mean_return'][t] = mean_return
            port_var['std_return'][t] = std_return
            port_var['sharp_ratio'][t] = sharp_ratio


def port_monte_carlo_stoch(simulations: str, start_year: int, start_month: int, end_year: int, scenario: str,
                           workspace_directory: str, input_directories: list[str], results_directory: str | None = None,
                           max_workers: int | None = None,
                           model_name: str = "monte_carlo", model_description: str = "Portfolio Monte Carlo simulation."):
    model = StochExecutor(name=model_name, description=model_description)
    model.bind_proj_func(port_monte_carlo_proj)
    model.set_run_config(
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
    valid, msg = vates.utils.validate_corr_matrix(corr_matrix)
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

    model.run(proj_func_args={
        "risk_free_rate": risk_free_rate,
        "n_assets": n_assets,
        "mu": mu,
        "sigma": sigma,
        "corr_matrix": corr_matrix,
        "portfolio_params": portfolio_params,
    })


def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    port_monte_carlo_stoch(**kwargs)


if __name__ == "__main__":
    main()

