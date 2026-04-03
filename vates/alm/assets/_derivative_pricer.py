import math
from scipy.stats import norm
from enum import Enum, unique

@unique
class LongOrShort(Enum):
    """Enum for long or short position."""
    LONG = "LONG"
    SHORT = "SHORT"

@unique
class CallOrPut(Enum):
    """Enum for call or put option."""
    CALL = "CALL"
    PUT = "PUT"


class BlackScholesCalculator:
    """
    Equity option Black-Scholes calculator.
    """
    @staticmethod
    def price(call_or_put: CallOrPut, s: float, k: float, r: float, q: float, sigma: float, tau: float) -> float:
        """
        Calculate Black-Scholes price.

        Args:
            call_or_put (CallOrPut): Call or put option.
            s (float): Stock price.
            k (float): Strike price.
            r (float): Risk-free interest rate (continuously compounded).
            q (float): Dividend yield assumption (continuously compounded).
            sigma (float): Standard deviation, i.e. volatility.
            tau (float): Time to expiration in years, i.e. `T - t`.

        Returns:
            float: Option price.
        """
        if tau < 0:
            raise ValueError(f'{tau=}, must be non-negative.')

        if tau < 1e-10:
            return max(s - k, 0.0) if call_or_put == CallOrPut.CALL else max(k - s, 0.0)

        f = s * math.exp((r - q) * tau)  # forward price: `F = S * exp((r - q) * t)`
        vol_tau = sigma * math.sqrt(tau)  # volatility over tau
        d1 = math.log(f / k) / vol_tau + 0.5 * vol_tau
        d2 = d1 - vol_tau
        nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
        z = math.exp(-r * tau)

        if call_or_put == CallOrPut.CALL:
            return z * (f * nd1 - k * nd2)
        else:
            return z * (k * (1 - nd2) - f * (1 - nd1))

    @staticmethod
    def greeks(call_or_put: CallOrPut, s: float, k: float, r: float, q: float, sigma: float, tau: float
               ) -> dict[str, float]:
        """
        Calculate Black-Scholes Greeks.

        Args:
            call_or_put (CallOrPut): Call or put option.
            s (float): Stock price.
            k (float): Strike price.
            r (float): Risk-free interest rate (continuously compounded).
            q (float): Dividend yield assumption (continuously compounded).
            sigma (float): Standard deviation, i.e. volatility.
            tau (float): Time to expiration in years, i.e. `T - t`.

        Returns:
            dict[str, float]: Dictionary of greeks.
        """
        if tau < 0: raise ValueError(f'{tau=}, must be non-negative.')
        if tau < 1e-10: return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0, }

        f = s * math.exp((r - q) * tau)  # forward price: `F = S * exp((r - q) * t)`
        vol_tau = sigma * math.sqrt(tau)  # volatility over tau
        d1 = math.log(f / k) / vol_tau + 0.5 * vol_tau
        d2 = d1 - vol_tau
        nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
        pdf_d1 = norm.pdf(d1)
        z, zq = math.exp(-r * tau), math.exp(-q * tau)

        if call_or_put == CallOrPut.CALL:
            delta = zq * nd1  # `exp(-q * tau) * N(d1)`
            theta = -0.5 * z * f * sigma * pdf_d1 / math.sqrt(tau) + z * (q * f * nd1 - r * k * nd2)
            # `-0.5 * exp(-q * tau) * S * sgima * N'(d1) / sqrt(tau) + q * exp(-q * tau) * S * N(d1) - r * exp(-r * tau) * K * N(d2)`
            rho = z * k * tau * nd2  # `exp(-r * tau) * K * tau * N(d2)
        else:
            delta = zq * (nd1 - 1)  # `-exp(-q * tau) * N(-d1)`
            theta = -0.5 * z * f * sigma * pdf_d1 / math.sqrt(tau) + z * (q * f * (nd1 - 1) - r * k * (nd2 - 1))
            # `-0.5 * exp(-q * tau) * S * sgima * N'(d1) / sqrt(tau) - q * exp(-q * tau) * S * N(-d1) + r * exp(-r * tau) * K * N(-d2)`
            rho = z * k * tau * (nd2 - 1)  # `-exp(-r * tau) * K * tau * N(-d2)

        gamma = zq * pdf_d1 / s / vol_tau  # `exp(-q * tau) * N'(d1) / (S * sigma * sqrt(tau))`
        vega = z * f * math.sqrt(tau) * pdf_d1  # `exp(-q * tau) * S * sqrt(tau) * N'(d1)`

        return {
            'delta': float(delta),  # `dP/dS`
            'gamma': float(gamma),  # `d2P/dS2`
            'theta': float(theta),  # `dP/dt`
            'vega': float(vega),  # `dP/dSigma
            'rho': float(rho),  # `dP/dr`
        }

    @staticmethod
    def _price_and_vega(call_or_put: CallOrPut, s: float, k: float, r: float, q: float, sigma: float, tau: float
                        ) -> tuple[float, float]:
        """
        Calculate Black-Scholes price and vega.

        Args:
            call_or_put (CallOrPut): Call or put option.
            s (float): Stock price.
            k (float): Strike price.
            r (float): Risk-free interest rate (continuously compounded).
            q (float): Dividend yield assumption (continuously compounded).
            sigma (float): Standard deviation, i.e. volatility.
            tau (float): Time to expiration in years, i.e. `T - t`.

        Returns:
            tuple[float, float]: Price and vega (`dPrice/dSigma`).
        """
        if tau < 0:
            raise ValueError(f'{tau=}, must be non-negative.')

        if tau < 1e-10:
            return max(s - k, 0.0) if call_or_put == CallOrPut.CALL else max(k - s, 0.0), 0.0

        f = s * math.exp((r - q) * tau)  # forward price: `F = S * exp((r - q) * t)`
        vol_tau = sigma * math.sqrt(tau)  # volatility over tau
        d1 = math.log(f / k) / vol_tau + 0.5 * vol_tau
        d2 = d1 - vol_tau
        nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
        z = math.exp(-r * tau)

        if call_or_put == CallOrPut.CALL:
            price = z * (f * nd1 - k * nd2)
        else:
            price = z * (k * (1 - nd2) - f * (1 - nd1))

        vega = z * f * norm.pdf(d1) * math.sqrt(tau)

        return price, vega

    @staticmethod
    def implied_volatility(call_or_put: CallOrPut, price: float, s: float, k: float, r: float, q: float, tau: float,
                           initial_guess: float = 0.2, tol: float = 1e-10, maxiter: int = 100) -> float:
        """
        Solve Black-Scholes implied volatility (sigma).

        Strategy:
          1) Validate price is within no-arbitrage bounds.
          2) Use Newton-Raphson starting from `initial_guess`.
          3) If NR fails to converge (or vega too small), fall back to bisection.

        Args:
            call_or_put (CallOrPut): Call or put option.
            price (float): Option price.
            s (float): Stock price.
            k (float): Strike price.
            r (float): Risk-free interest rate (continuously compounded).
            q (float): Dividend yield assumption (continuously compounded).
            tau (float): Time to expiration in years, i.e. `T - t`.
            initial_guess (float): Initial guess for sigma (default 0.2)
            tol (float): Convergence tolerance on sigma (absolute)
            maxiter (int): Maximum NR iterations before switching to bisection

        Returns:
            float: Implied volatility.
        """
        _price_and_vega = BlackScholesCalculator._price_and_vega
        # --- handle immediate expiry ---
        if tau <= 1e-12:
            # At expiry price must equal intrinsic (discounting irrelevant since tau ~ 0)
            intrinsic = max(s - k, 0.0) if call_or_put == CallOrPut.CALL else max(k - s, 0.0)
            # For numerical robustness allow tiny tolerance
            if abs(price - intrinsic) <= 1e-12:
                return 0.0
            raise ValueError("Price inconsistent with immediate-expiry intrinsic value")

        # --- no-arbitrage price bounds (discounted forward form) ---
        disc = math.exp(-r * tau)
        f = s * math.exp((r - q) * tau)

        if call_or_put == CallOrPut.CALL:
            lower_bound = max(0.0, disc * (f - k))  # = exp(-rτ) * max(F - K, 0)
            upper_bound = disc * f  # = S * exp(-qτ)
        else:
            lower_bound = max(0.0, disc * (k - f))
            upper_bound = disc * k  # upper bound for put is K*exp(-rτ)

        # Allow tiny epsilon for market microstructure/pricing noise
        eps = 1e-12
        if price < lower_bound - eps or price > upper_bound + eps:
            raise ValueError(f"Price {price} outside no-arbitrage bounds [{lower_bound:.12g}, {upper_bound:.12g}]")

        # If price equals intrinsic (discounted) within tolerance, implied vol is 0
        if abs(price - lower_bound) <= 1e-14:
            return 0.0

        # --- Newton-Raphson ---
        sigma = max(1e-12, initial_guess)
        for i in range(maxiter):
            p, v = _price_and_vega(call_or_put, s, k, r, q, sigma, tau)
            diff = p - price
            if abs(diff) < 1e-12:
                return max(0.0, sigma)
            # If vega is tiny, break to bisection
            if v < 1e-14:
                break
            update = diff / v
            sigma -= update
            # keep sigma positive
            if sigma <= 0:
                sigma = 1e-12
            if abs(update) < tol:
                return max(0.0, sigma)

        # --- Bisection fallback (robust) ---
        # Find bracket [lo, hi] such that price(lo) - market_price and price(hi) - market_price have opposite signs
        lo = 1e-12
        hi = 1.0  # start with 100% vol as upper, expand if necessary
        p_lo, _ = _price_and_vega(call_or_put, s, k, r, q, lo, tau)
        p_hi, _ = _price_and_vega(call_or_put, s, k, r, q, hi, tau)

        # expand hi until sign change or until hi becomes huge
        expand_iter = 0
        while (p_lo - price) * (p_hi - price) > 0 and expand_iter < 50:
            hi *= 2
            p_hi, _ = _price_and_vega(call_or_put, s, k, r, q, hi, tau)
            expand_iter += 1

        if (p_lo - price) * (p_hi - price) > 0:
            # cannot bracket — should not generally happen because price is within bounds
            raise RuntimeError("Failed to bracket implied volatility")

        # now bisection until tolerance achieved
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            p_mid, _ = _price_and_vega(call_or_put, s, k, r, q, mid, tau)
            if abs(p_mid - price) < 1e-12 or (hi - lo) < tol:
                return max(0.0, mid)
            # decide side
            if (p_lo - price) * (p_mid - price) <= 0:
                hi = mid
                # p_hi = p_mid
            else:
                lo = mid
                p_lo = p_mid

        # If we reach here, return midpoint (the best effort)
        return max(0.0, 0.5 * (lo + hi))
