import numpy as np
import pandas as pd
import math

import vates as vt

class PortMonteCarlo(vt.ProjModelEngine):
    """Portfolio Monte Carlo simulation."""
    def __init__(self, risk_free_rate, asset_name_list, n_assets, mu, sigma, corr_matrix, portfolios, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_free_rate: float = risk_free_rate
        self.m_rf_rate: float = (1 + self.risk_free_rate) ** (1/12) - 1
        self.asset_name_list: list = asset_name_list.copy()
        self.n_assets: int = n_assets
        self.mu: np.ndarray = mu.copy()  # continually compounded, i.e. ln(1 + annaul return)
        self.sigma: np.ndarray = sigma.copy()
        self.corr_matrix: np.ndarray = corr_matrix.copy()
        self.portfolios: dict[str, dict] = portfolios.copy()

        for port_name, port_var in self.portfolios.items():
            port_var['asset_amount'] = port_var['init_balance'] * port_var['weight']
            port_var['rfa_amount'] = port_var['init_balance'] * port_var['rfawgt']
            port_var['balance'] = vt.TDepVariable(self,"balance", port_name, 'MonteCarlo')
            port_var['return'] = vt.TDepVariable(self, "return", port_name, 'MonteCarlo')
            port_var['sum_ret'] = 0.0
            port_var['sum_sqret'] = 0.0

        seed = self.SIMULATION
        self.rng = np.random.default_rng(seed)

    def time_zero_calculations(self):
        p = self.period
        for _, port_var in self.portfolios.items():
            port_var['balance'][p] = port_var['init_balance']

    def in_time_calculations(self):
        p = self.period
        dt = 1 / 12

        if self.n_assets > 1:
            z = vt.utils.multivariate_standard_normal(self.corr_matrix, rng=self.rng, bypass_valid_corr=True)
        else:
            z = self.rng.standard_normal(1)  # note: standard_normal(1) >> np.ndarray, standard_normal(0) >> float

        r = np.array([vt.utils.geometric_brownian_motion(mu=self.mu[i], sigma=self.sigma[i], dt=dt, z=z[i])
                      for i in range(self.n_assets)]) - 1  # # S(t)/S(t-1) - 1

        for _, port_var in self.portfolios.items():
            port_var['asset_amount'] += port_var['asset_amount'] * r
            port_var['rfa_amount'] += port_var['rfa_amount'] * self.m_rf_rate
            port_var['balance'][p] = sum(port_var['asset_amount']) + port_var['rfa_amount']
            ret = port_var['balance'][p] / port_var['balance'][p - 1] - 1
            port_var['return'][p] = ret
            port_var['sum_ret'] += ret
            port_var['sum_sqret'] += ret ** 2

            if p.month % (12 // port_var['rebalance_freq']) == 0:  # reblance at this month
                port_var['asset_amount'] = port_var['balance'][p] * port_var['weight']
                port_var['rfa_amount'] = port_var['balance'][p] * port_var['rfawgt']

    def post_time_calculations(self):
        end_date = pd.Period(f'{self.END_YEAR}-12', freq='M')
        n_months = (self.END_YEAR * 12 + 12) - (self.START_YEAR * 12 + self.START_MONTH)

        for port_name, port_var in self.portfolios.items():
            twrr = port_var['balance'][end_date] ** (12 / n_months) - 1  # time-weighted rate of return (TWRR)
            mean_return = port_var['sum_ret'] / n_months
            ms_return = port_var['sum_sqret'] / n_months  # mean squared return
            std_return = math.sqrt(ms_return - mean_return ** 2)  # variance = E(X^2) - (E(X))^2
            mean_return = (1 + mean_return) ** 12 - 1  # annualized
            std_return = std_return * math.sqrt(12)  # annualized
            sharp_ratio = (mean_return - self.risk_free_rate) / std_return

            port_var['twrr'] = vt.ConstVariable(self, 'twrr', port_name, 'MonteCarlo')
            port_var['mean_return'] = vt.ConstVariable(self, 'mean_return', port_name, 'MonteCarlo')
            port_var['std_return'] = vt.ConstVariable(self, 'std_return', port_name, 'MonteCarlo')
            port_var['sharp_ratio'] = vt.ConstVariable(self, 'sharp_ratio', port_name, 'MonteCarlo')

            port_var['twrr'][0] = twrr
            port_var['mean_return'][0] = mean_return
            port_var['std_return'][0] = std_return
            port_var['sharp_ratio'][0] = sharp_ratio

class PortMonteCarloStoch(vt.StochExecutor):

    def pre_stoch_calculations(self):
        negative_weight_allowed = False
        negatvie_rfawgt_allowed = False
        # parameters
        df = self.read_csv('parameters', index_col="parameter")
        risk_free_rate = df.at['risk_free_rate', 'value']
        # assets
        df = self.read_csv('assets', index_col="asset_name")
        asset_name_list = df.index.tolist()
        n_assets = len(asset_name_list)
        mu, sigma, corr_matrix = np.zeros(n_assets), np.zeros(n_assets), np.zeros(shape=(n_assets, n_assets))
        for i in range(n_assets):
            asset_name = asset_name_list[i]
            mu[i] = math.log(1 + df.at[asset_name, 'expected_return'])  # convert to continually compounded
            sigma[i] = df.at[asset_name, 'standard_deviation']
            for j in range(n_assets):
                corr_matrix[i, j] = df.at[asset_name, asset_name_list[j]]
        valid, msg = vt.utils.validate_corr_matrix(corr_matrix)
        if not valid: raise ValueError(msg)

        # portfolios
        df = self.read_csv('portfolios', index_col="portfolio_name")
        portfolios: dict[str, dict] = {}
        for port in df.index:
            init_balance = df.at[port, 'initial_balance']
            rebalance_freq = df.at[port, 'rebalance_freq']
            weight = np.array([df.at[port, asset_name_list[i]] for i in range(n_assets)])
            rfawgt = 1 - sum(weight)
            if not negative_weight_allowed and any(weight < 0):
                raise ValueError(f'Negative asset weight is NOT allowed. Please revise input of portfolio {port}.')
            if not negatvie_rfawgt_allowed and rfawgt < 0:
                raise ValueError(f'Sum of asset weight exceeds 100%. Please revise input of portfolio {port}.')
            portfolios[port] = {
                'init_balance': init_balance,
                'rebalance_freq': rebalance_freq,
                'weight': weight,
                'rfawgt': rfawgt,
            }

        self.ALL_SIM_PARAMS = {
            'risk_free_rate': risk_free_rate,
            'asset_name_list': asset_name_list,
            'n_assets': n_assets,
            'mu': mu,
            'sigma': sigma,
            'corr_matrix': corr_matrix,
            'portfolios': portfolios,
        }

    def post_stoch_calculations(self):
        pass

if __name__ == "__main__":
    vt.cli.run_model(PortMonteCarlo, stoch_cls=PortMonteCarloStoch)
