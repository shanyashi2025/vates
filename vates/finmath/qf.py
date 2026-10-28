"""Quantitative Finance"""
import numpy as np
import warnings


_ENGINES = ('cvxpy', 'scipy')
# `scipy` (SLSQP) is much faster for a small frontier but is a local solver and can settle
# in a local minimum when there are many assets; `cvxpy` solves each convex QP globally.
# The crossover measured on the example workspace is around 25-50 assets.
_MAX_ASSETS_FOR_SCIPY = 20


def multivariate_standard_normal(corr: np.ndarray, rng: np.random._generator.Generator = None,
                                 bypass_valid_corr: bool = False) -> np.ndarray:
    """Draw a standard normal vector with the given correlation matrix.

    Args:
        corr (np.ndarray): Correlation matrix (n x n).
        rng (np.random.Generator, optional): Random generator. Defaults to `np.random.default_rng()`.
        bypass_valid_corr (bool): Skip correlation-matrix validation, defaults to False.

    Returns:
        np.ndarray: Correlated standard normal vector of length n.

    Raises:
        ValueError: If `corr` is not a valid correlation matrix.
    """
    if not bypass_valid_corr:
        valid, msg = validate_corr_matrix(corr)
        if not valid:
            raise ValueError(msg)
    rng = rng or np.random.default_rng()
    n = len(corr)
    L = np.linalg.cholesky(corr)
    u = rng.standard_normal(n)
    z = L @ u
    return z


def validate_corr_matrix(corr_matrix: np.ndarray, psd_tol: float = 1e-10) -> tuple[bool, str]:
    """Validate that an array is a usable correlation matrix.

    Checks shape, symmetry, unit diagonal, off-diagonal entries within [-1, 1], and
    positive semi-definiteness.

    Args:
        corr_matrix (np.ndarray): Candidate correlation matrix (n x n).
        psd_tol (float): Tolerance for the smallest eigenvalue; a matrix whose smallest
            eigenvalue is below `-psd_tol` is rejected. Defaults to 1e-10.

    Returns:
        tuple[bool, str]: `(True, 'pass')` if valid, otherwise `(False, reason)`.

    Notes:
        Positive semi-definiteness is required for a correlation matrix to be realizable.
        A matrix passing the symmetry and range checks can still have a negative eigenvalue
        (e.g. three assets all pairwise correlated at -0.9); such a matrix is rejected here
        rather than failing later, e.g. as `nan` variances or an `np.linalg.cholesky` error.
    """
    if type(corr_matrix) != np.ndarray:
        return False, f"Invalid {type(corr_matrix)=}, expected np.ndarray."
    shape = corr_matrix.shape
    if corr_matrix.ndim != 2:
        return False, f"Invalid {shape=}, expected n x n."
    if shape[0] != shape[1]:
        return False, f"Invalid {shape=}, expected n x n."
    n = shape[0]
    if n == 0:
        return False, f"Invalid {shape=}, expected at least one asset."
    for i in range(n):
        if corr_matrix[i, i] != 1:
            return False, f"Invalid rho[{i + 1},{i + 1}]={float(corr_matrix[i, i])}, expected =1."
        for j in range(i + 1, n):
            if not -1 <= corr_matrix[i, j] <= 1:
                return False, f"Invalid rho[{i + 1},{j + 1}]={float(corr_matrix[i, j])}, expected -1 to 1."
            if corr_matrix[i, j] != corr_matrix[j, i]:
                return False, (f"rho[{i + 1},{j + 1}]={float(corr_matrix[i, j])}, "
                               f"rho[{j + 1},{i + 1}]={float(corr_matrix[j, i])}, expected equal")
    # every principal 2x2 minor is PSD once diagonals are 1 and |rho| <= 1, so only n >= 3 can fail here
    if n >= 3:
        # eigvalsh is symmetric-only; round-off in the stored values has already been accepted above
        M = (corr_matrix + corr_matrix.T) / 2.0
        if not np.all(np.isfinite(M)):
            return False, "Invalid rho contains non-finite values (nan/inf)."
        min_eig = float(np.linalg.eigvalsh(M)[0])
        if min_eig < -psd_tol:
            return False, f"Invalid rho: matrix is not positive semi-definite, {min_eig=:.6e}."
    return True, 'pass'


def geometric_brownian_motion(mu: float, sigma: float, dt: float, z: np.ndarray) -> np.ndarray:
    """Geometric Brownian Motion growth factor.

    Args:
        mu (float): Drift.
        sigma (float): Volatility.
        dt (float): Time step.
        z (np.ndarray): Standard normal shock(s).

    Returns:
        np.ndarray: Growth factor(s) `exp((mu - 0.5 * sigma ** 2) * dt + sigma * z * sqrt(dt))`.
    """
    return np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * z * np.sqrt(dt))


def search_efficient_frontier(mu: np.ndarray, sigma: np.ndarray, corr_matrix: np.ndarray, n_points: int = 100,
                              engine: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace the efficient frontier of a long-only, fully-invested portfolio set.

    For each target return between `min(mu)` and `max(mu)`, minimizes the portfolio variance
    subject to `sum(w) = 1`, `w >= 0` and `mu @ w = r_target`.

    Args:
        mu (np.ndarray): Expected returns, length n.
        sigma (np.ndarray): Standard deviations, length n, non-negative.
        corr_matrix (np.ndarray): Correlation matrix (n x n), must be positive semi-definite.
        n_points (int): Number of target returns, defaults to 100; non-positive values are
            reset to 100 and values above 200 trigger a warning.
        engine (str | None): Backend to use, one of `'cvxpy'` or `'scipy'`. Defaults to `None`,
            which picks `'scipy'` when there are at most `_MAX_ASSETS_FOR_SCIPY` assets and
            `'cvxpy'` otherwise. If the preferred backend is not installed, the other one is
            used with a warning; if neither is installed, `ImportError` is raised.
            Invalid values raise `TypeError`/`ValueError`; an explicit choice that is not
            installed raises `ImportError` instead of falling back.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: `(wgts, rets, stds)` where `wgts` is an
            `(m, n)` array of portfolio weights, `rets` the `m` target returns, and `stds` the
            corresponding `m` standard deviations. `m <= n_points`: targets the backend could
            not solve are omitted.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the arguments are inconsistent or `corr_matrix` is invalid,
            or if `engine` is not one of the supported backends.
        ImportError: If no supported optimization backend is installed.

    Notes:
        - `'scipy'` (SLSQP) is much faster for a small frontier but is a local solver,
          so on many assets it can miss the global minimum variance.
        - `'cvxpy'` solves each convex QP globally and is preferred for large n.
        - If an explicit `engine` is requested and not installed, `ImportError` is raised
          rather than silently switching backends.
    """
    if engine is not None and type(engine) != str:
        raise TypeError(f'Invalid {type(engine)=}, expected None or str.')
    if engine is not None and engine not in _ENGINES:
        raise ValueError(f'Invalid {engine=}, expected None or one of {_ENGINES}.')
    n_points = _validate_efficient_frontier_args(mu, sigma, corr_matrix, n_points)

    if engine is None:
        preferred = 'scipy' if len(mu) <= _MAX_ASSETS_FOR_SCIPY else 'cvxpy'
        other = 'cvxpy' if preferred == 'scipy' else 'scipy'
        if _backend_available(preferred):
            engine = preferred
        elif _backend_available(other):
            engine = other
            warnings.warn(f"Backend '{preferred}' is not installed, falling back to '{other}'.")
        else:
            raise ImportError("Need to install 'scipy' (`pip install scipy`) or 'cvxpy' "
                              "(`pip install cvxpy`, with a QP solver such as CLARABEL).")

    if engine == 'cvxpy':
        return _search_efficient_frontier_cvxpy(mu, sigma, corr_matrix, n_points)
    return _search_efficient_frontier_scipy(mu, sigma, corr_matrix, n_points)


def _backend_available(engine: str) -> bool:
    """Check whether an optimization backend (and, for cvxpy, a QP solver) can be imported.

    Imports are cached by Python, so this costs nothing extra on repeated calls.
    """
    try:
        if engine == 'cvxpy':
            import cvxpy as cp
            # `pip install cvxpy` alone does not guarantee a QP-capable solver is present
            return bool({'CLARABEL', 'OSQP', 'ECOS', 'SCS'} & set(cp.installed_solvers()))
        from scipy.optimize import minimize  # noqa: F401
    except Exception:  # missing package, or a broken install (e.g. missing binary)
        return False
    return True


def _search_efficient_frontier_cvxpy(mu: np.ndarray, sigma: np.ndarray, corr_matrix: np.ndarray, n_points: int
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`search_efficient_frontier` using `cvxpy` (global optimum per target).

    Arguments must already be validated.
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("Need to install 'cvxpy' library (`pip install cvxpy`), "
                          "see https://www.cvxpy.org/install/ for the required solver.") from None

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


def _search_efficient_frontier_scipy(mu: np.ndarray, sigma: np.ndarray, corr_matrix: np.ndarray, n_points: int
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`search_efficient_frontier` using `scipy.optimize.minimize` (SLSQP).

    A local solver: fast for a small frontier, but it may settle in a local minimum when
    there are many assets. Arguments must already be validated.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError("Need to install 'scipy' library (`pip install scipy`).") from None

    cov_matrix = corr_matrix * np.outer(sigma, sigma)
    n = len(mu)
    # long-only, fully-invested simplex: sum(w) = 1, w >= 0
    bounds = [(0.0, 1.0)] * n
    sum_c = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(n)}
    x0 = np.full(n, 1.0 / n)

    def var_grad(w):  # gradient of w' C w
        return 2.0 * (cov_matrix @ w)

    def quad_form(w):
        return float(w @ cov_matrix @ w)

    wgts, rets, stds = [], [], []  # to store solutions
    # warm-start from the previous (nearby) solution to speed up convergence
    last_w = None
    # with (near-)constant `mu` the return constraint is linearly dependent on sum(w)=1,
    # which makes SLSQP's LSQ subproblem singular; dropping it is exact in that case.
    ret_redundant = float(np.ptp(mu)) <= 1e-12 * max(1.0, float(np.abs(mu).max()))

    # target returns from min to max
    for r_target in np.linspace(mu.min(), mu.max(), n_points):
        ret_c = {'type': 'eq', 'fun': lambda w, r=r_target: mu @ w - r,
                 'jac': lambda w, r=r_target: mu}
        cons = [sum_c] if ret_redundant else [sum_c, ret_c]
        res = minimize(quad_form, last_w if last_w is not None else x0, jac=var_grad,
                       method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-12, 'maxiter': 500})
        if not res.success:
            warnings.warn(f'Optimization failed for {r_target=}: {res.message}')
            continue

        # SLSQP works in float tolerance; clip tiny negatives and renormalize onto the simplex
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        last_w = w
        wgts.append(w)
        stds.append(np.sqrt(quad_form(w)))
        rets.append(r_target)

    return np.array(wgts), np.array(rets), np.array(stds)


def _validate_efficient_frontier_args(mu: np.ndarray, sigma: np.ndarray, corr_matrix: np.ndarray, n_points: int
                                      ) -> int:
    """Validate and normalize arguments shared by the efficient frontier implementations."""
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
        warnings.warn(f'n_points is set to 100 to prevent crash, reason: {n_points=}, expceted > 0.')
        n_points = 100
    elif n_points > 200:
        warnings.warn(f'{n_points=}, consider reduce number of points?')
    return n_points
