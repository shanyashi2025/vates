"""Quantitative Finance and Investment"""
import numpy as np
import math
import warnings


def multivariate_standard_normal(corr: np.ndarray, rng: np.random._generator.Generator=None,
                                 bypass_valid_corr: bool=False) -> np.ndarray:
    if not bypass_valid_corr: validate_corr_matrix(corr)
    if rng is None: rng = np.random.default_rng()
    n = len(corr)
    L = np.linalg.cholesky(corr)
    u = rng.standard_normal(n)
    z = L @ u
    return z


def validate_corr_matrix(corr_matrix: np.ndarray) -> tuple[bool, str]:
    if type(corr_matrix) != np.ndarray:
        return False, f"Invalid {type(corr_matrix)=}, expected np.ndarray."
    shape = corr_matrix.shape
    if len(shape) != 2:
        return False, f"Invalid {len(shape)=}, expected 2."
    if shape[0] != shape[1]:
        return False, f"Invalid {shape=}, expected n x n."
    n = shape[0]
    for i in range(n):
        if corr_matrix[i, i] != 1:
            return False, f"Invalid rho[{i + 1},{i + 1}]={float(corr_matrix[i, i])}, expected =1."
        for j in range(i + 1, n):
            if not -1 <= corr_matrix[i, j] <= 1:
                return False, f"Invalid rho[{i + 1},{j + 1}]={float(corr_matrix[i, j])}, expected -1 to 1."
            if corr_matrix[i, j] != corr_matrix[j, i]:
                return False, (f"rho[{i + 1},{j + 1}]={float(corr_matrix[i, j])}, "
                               f"rho[{j + 1},{i + 1}]={float(corr_matrix[j, i])}, expected equal")
    return True, 'pass'


def geometric_brownian_motion(mu, sigma, dt, z):
    """Geometric Brownian Motion"""
    return math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * z * math.sqrt(dt))


def search_efficient_frontier(mu: np.ndarray, sigma: np.ndarray, corr_matrix: np.ndarray, n_points: int=100
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cvxpy as cp
    # validate arguments
    if type(mu) != np.ndarray: raise TypeError(f'Invalid {type(mu)=}, expected np.ndarray.')
    if type(sigma) != np.ndarray: raise TypeError(f'Invalid {type(sigma)=}, expected np.ndarray.')
    if type(corr_matrix) != np.ndarray: raise TypeError(f'Invalid {type(corr_matrix)=}, expected np.ndarray.')
    if type(n_points) != int: raise TypeError(f'Invalid {type(n_points)=}, expected int.')
    if len(mu) != len(sigma): raise ValueError(f'{len(mu)=}, {len(sigma)=}, expected identical.')
    if corr_matrix.ndim != 2: raise ValueError(f'{corr_matrix.ndim=}, expected 2.')
    if any(sigma < 0): raise ValueError(f'sigma must be non-negative.')
    valid , msg = validate_corr_matrix(corr_matrix)
    if not valid: raise ValueError(msg)
    if corr_matrix.shape[0] != len(mu):
        raise ValueError(f'shape of corr_matrix {corr_matrix.shape} not consistent with len of mu and sigma {len(mu)}.')
    if n_points <= 0:
        n_points = 100
        warnings.warn(f'n_points is set to 100 to prevent crash, reason: {n_points=}, expceted > 0.')
    elif n_points > 200:
        warnings.warn(f'{n_points=}, consider reduce number of points?')

    cov_matrix = corr_matrix * np.outer(sigma, sigma)
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    std = cp.quad_form(w, cov_matrix)

    wgts, rets, stds = [], [], []  # to store solutions

    # target returns from min to max
    for r_target in np.linspace(mu.min(), mu.max(), n_points):
        prob = cp.Problem(cp.Minimize(std),[cp.sum(w) == 1, w >= 0, ret == r_target])
        prob.solve()

        if w.value is not None:
            wgts.append(w.value)
            stds.append(np.sqrt(w.value.T @ cov_matrix @ w.value))
            rets.append(r_target)

    return np.array(wgts), np.array(rets), np.array(stds)
