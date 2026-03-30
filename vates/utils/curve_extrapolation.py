import numpy as np
from typing import Dict

class SmithWilsonExtrapolator:
    """Smith–Wilson curve extrapolator for risk-free interest rates.

    References:
    - EIOPA (7 Dec 2015) Risk-free interest rate technical documentation.
    """

    def __init__(self,
                 instrument: str,
                 coupon_frequency: int,
                 rates: np.ndarray,
                 maturities: np.ndarray,
                 ufr: float,
                 convergence_point: int,
                 min_alpha: float,
                 convergence_tolerance_bp: float):
        """Initialize the Smith–Wilson extrapolator.

        Args:
            instrument: Instrument type: "zero", "bond", or "swap" (case-insensitive).
            coupon_frequency: Number of coupon payments per year for coupon
                instruments (1, 2, 4, or 12). Ignored for zeros.
            rates: Observed market rates as decimals. For bonds/swaps, use
                coupon rates or par rates per instrument.
            maturities: Maturities corresponding to `rates` in years. For
                bonds/swaps, provide payment times in years for each quoted
                instrument.
            ufr: Ultimate forward rate as an annually compounded rate
                (decimal), e.g., 0.035 for 3.5%.
            convergence_point: Convergence point (in years) by which the forward
                rate is within the specified tolerance of the UFR.
            min_alpha: Minimum value of the alpha parameter used as a lower
                bound in calibration. Default is 0.05.
            convergence_tolerance_bp: Convergence tolerance in basis points (bp)
                used as tau in calibration.

        Raises:
            ValueError: If `rates` and `maturities` lengths differ.
            ValueError: If `instrument` is not one of {"zero", "bond", "swap"}.
            ValueError: If `coupon_frequency` is not in {1, 2, 4, 12}.
        """
        if len(rates) != len(maturities):
            raise ValueError("Market rates and maturities must have same length")
        if instrument.lower() not in ("zero", "bond", "swap"):
            raise ValueError(f"Invalid instrument: {instrument}.")
        if int(coupon_frequency) not in (1, 2, 4, 12):
            raise ValueError(f"Invalid coupon frequency: {coupon_frequency}.")

        self.instrument = instrument.lower()
        self.r = rates
        self.u = maturities
        self.ln_ufr = np.log(1 + ufr)
        self.t2 = convergence_point
        self.n_coup = 1 if instrument.lower() == "zero" else int(coupon_frequency)
        self.n_rates = len(self.u)
        self.n_pays = int(max(self.u)) * self.n_coup

        self.alfa, self.gamma = self._calibrate_alfa_gamma(alfamin=min_alpha, tau=convergence_tolerance_bp/10000)

    def extrapolate(self, max_maturity, alfa=None, gamma=None):
        """Extrapolate the curve up to a maximum maturity.

        Computes discount factors, zero and forward rate intensities, and
        annual-compounded/continuously-compounded zero and forward rates from
        t=0 to `max_maturity` years, consistent with the Smith–Wilson model
        calibrated at initialization.

        Args:
            max_maturity: Maximum maturity in whole years (inclusive) for which
                the outputs are returned.
            alfa: Alpha value.
            gamma: Vector Q b that parameterizes the kernel representation.

        Returns:
            Dict[str, np.ndarray]: A dictionary with arrays indexed by maturity:
                - "discount": Discount factors.
                - "yldintensity": Zero rate intensity (log-yield) at each t.
                - "zeroac": Annual-compounded zero rates.
                - "fwintensity": Instantaneous forward rate intensity.
                - "forwardcc": Continuously-compounded one-year forward rates.
                - "forwardac": Annual-compounded one-year forward rates.
        """
        if alfa is None: alfa = self.alfa
        if gamma is None: gamma = self.gamma

        v = np.arange(max_maturity + 1).reshape(-1, 1)
        u = np.arange(1, self.n_pays + 1).reshape(1, -1) / self.n_coup

        H = self._H_mat(alfa, v, u)
        G = self._G_mat(alfa, v, u)

        tempdiscount = (H @ gamma).flatten()  # spec 153, H(v,u)*Qb
        tempintensity = (G @ gamma).flatten()  # spec 157, G(v,u)*Qb

        discount = np.zeros(max_maturity + 1)
        yldintensity = np.zeros(max_maturity + 1)
        fwintensity = np.zeros(max_maturity + 1)
        zeroac = np.zeros(max_maturity + 1)
        forwardac = np.zeros(max_maturity + 1)
        forwardcc = np.zeros(max_maturity + 1)
        zerocc = np.zeros(max_maturity + 1)

        discount[0] = 1
        temp = np.sum((1 - np.exp(-alfa * u.flatten())) * gamma.flatten())  # spec 159, (1'-exp(-alfa*u'))Qb
        yldintensity[0] = self.ln_ufr - alfa * temp  # spec 159
        fwintensity[0] = yldintensity[0]  # spec 159

        maturities = np.arange(1, max_maturity + 1)
        discount[1:] = np.exp(-self.ln_ufr * maturities) * (1 + tempdiscount[1:])
        yldintensity[1:] = self.ln_ufr - np.log1p(tempdiscount[1:]) / maturities  # spec 157
        fwintensity[1:] = self.ln_ufr - tempintensity[1:] / (1 + tempdiscount[1:])  # spec 157
        zeroac[1:] = discount[1:] ** (-1 / maturities) - 1
        forwardac[1:] = discount[:-1] / discount[1:] - 1
        forwardcc[1:] = np.log1p(forwardac[1:])
        zerocc[1:] = np.log1p(zeroac[1:])

        return {
            "discount": discount,
            "yldintensity": yldintensity,
            "zeroac": zeroac,
            "fwintensity": fwintensity,
            "forwardcc": forwardcc,
            "forwardac": forwardac
        }

    def _C_mat(self) -> np.ndarray:
        """Build the cash-flow matrix for the selected instrument set.

        Returns:
            np.ndarray: Matrix of shape (number_of_payment_times, number_of_quotes)
            mapping quoted instruments to unit cash flows at each payment time.
            Zeros have a single payment; bonds and swaps include coupons and
            principals according to `coupon_frequency`.
        """
        C = np.zeros((self.n_pays, self.n_rates))

        if self.instrument == "zero":
            j = self.u.astype(int) - 1
            i = np.arange(self.n_rates)
            C[j, i] = (1 + self.r) ** self.u
        else:  # "bond" or "swap"
            for i in range(self.n_rates):
                j = np.arange(int(self.u[i] * self.n_coup))
                C[j, i] = self.r[i] / self.n_coup
                C[j[-1], i] += 1

        return C

    @staticmethod
    def _H_mat(alfa, u, v):
        """Compute the Smith–Wilson kernel H(alpha; u, v).

        Args:
            alfa: Alpha parameter (> 0).
            u: Column vector (or broadcastable) of times u in years.
            v: Row vector (or broadcastable) of times v in years.

        Returns:
            np.ndarray: Kernel matrix H evaluated at all (u, v) pairs.
        """
        u = alfa * u
        v = alfa * v

        def _hh(z): return (z + np.exp(-z)) / 2

        return _hh(u + v) - _hh(np.abs(u - v))  # spec 138

    @staticmethod
    def _G_mat(alfa, u, v):
        """Compute the derivative kernel G(alpha; u, v).

        Args:
            alfa: Alpha parameter (> 0).
            u: Column vector (or broadcastable) of times u in years.
            v: Row vector (or broadcastable) of times v in years.

        Returns:
            np.ndarray: Kernel matrix G evaluated at all (u, v) pairs.
        """
        return np.where(
            u > v,
            alfa * (1 - np.exp(-alfa * u) * np.cosh(alfa * v)),
            alfa * np.exp(-alfa * v) * np.sinh(alfa * u)
        )  # spec 141

    def _calibrate_alfa_gamma(self, alfamin, tau):
        """Calibrate alpha and gamma coefficients to match quotes and UFR.

        The calibration finds the smallest alpha ≥ `alfamin` such that
        g(alpha) ≤ tau, where g(alpha) measures proximity of the forward rate
        at `convergence_t` to the UFR within `tau`. It simultaneously solves
        for gamma (i.e., Q b) to match instrument prices.

        Args:
            alfamin: Lower bound for alpha during the search.
            tau: Convergence tolerance in decimal (e.g., 1 bp = 0.0001).

        Returns:
            Tuple[float, np.ndarray]: Calibrated alpha and gamma vector.
        """
        C = self._C_mat()
        d = np.exp(-self.ln_ufr * np.arange(1, self.n_pays + 1) / self.n_coup)  # spec 144
        Q = np.diag(d) @ C  # spec 145, Q = dΔQ

        galfa, gamma = self._galfa(alfamin, Q)
        if galfa <= tau:
            alfa = alfamin
        else:
            precision = 5
            stepsize = 0.1
            alfa = alfamin + stepsize
            while alfa <= 20:
                galfa, _ = self._galfa(alfa, Q)
                if galfa <= tau:
                    break
                alfa += stepsize
            for _ in range(precision):
                alfa, gamma = self._alfa_scan(alfa, gamma, stepsize, Q, tau)
                stepsize /= 10

        return alfa, gamma

    def _galfa(self, alfa, Q):
        """Compute g(alpha) and the gamma vector for a given alpha.

        Args:
            alfa: Candidate alpha value.
            Q: Cash-flow matrix adjusted by discount vector (d) with shape
                (number_of_payment_times, number_of_quotes).

        Returns:
            Tuple[float, np.ndarray]:
                - g(alpha): Convergence metric; lower is closer to UFR by `t2`.
                - gamma: Vector Q b that parameterizes the kernel representation.

        Notes:
            Follows Smith–Wilson specs: solves b = ((Q'HQ)^(-1))(p - q),
            gamma = Q b, then computes kappa and g(alpha) as in the spec.
        """
        u = np.arange(1, self.n_pays + 1) / self.n_coup
        H = self._H_mat(alfa, u.reshape(-1, 1), u.reshape(1, -1))

        temp1 = (1 - Q.sum(axis=0)).reshape(-1, 1)  # spec 145, (p - q) = (1 - Q'1), i.e. p = 1, q = Q'1
        b = np.linalg.inv(Q.T @ H @ Q) @ temp1  # spec 155, b = ((Q'HQ)^(-1))(p - q)
        gamma = Q @ b  # gamma = Qb

        temp2 = np.sum(gamma.flatten() * u)
        temp3 = np.sum(gamma.flatten() * np.sinh(alfa * u))
        kappa = (1 + alfa * temp2) / temp3  # spec 161

        galfa = alfa / abs(1 - kappa * np.exp(self.t2 * alfa))  # spec 164

        return galfa, gamma

    def _alfa_scan(self, lastalfa, lastgamma, stepsize, Q, tau):
        """Refine the alpha search by scanning in decreasing step sizes.

        Args:
            lastalfa: Current alpha value.
            lastgamma: Current gamma vector associated with `lastalfa`.
            stepsize: Current step size for the scan.
            Q: Discounted cash-flow matrix.
            tau: Target threshold for g(alpha).

        Returns:
            Tuple[float, np.ndarray]: Updated alpha and corresponding gamma.
        """
        alfa = lastalfa
        gamma = lastgamma
        for a in np.arange(lastalfa + stepsize / 10 - stepsize, lastalfa + 1e-12, stepsize / 10):
            galfa, gamma = self._galfa(a, Q)
            if galfa <= tau:
                alfa = a
                break
        return alfa, gamma


def smith_wilson_extrap(rates: np.ndarray,
                        maturities: np.ndarray,
                        ufr: float,
                        convergence_point: int,
                        min_alpha: float,
                        instrument: str = "zero",
                        coupon_frequency: int = 1,
                        convergence_tolerance_bp: float = 0.1,
                        max_maturity: int = 150) -> Dict[str, np.ndarray]:
    """Smith-Wilson Risk-Free Interest Rate Extrapolation Tool.

    Args:
        instrument: Instrument type: "zero", "bond", or "swap" (case-insensitive).
        coupon_frequency: Number of coupon payments per year for coupon
            instruments (1, 2, 4, or 12). Ignored for zeros.
        rates: Observed market rates as decimals. For bonds/swaps, use
            coupon rates or par rates per instrument.
        maturities: Maturities corresponding to `rates` in years. For
            bonds/swaps, provide payment times in years for each quoted
            instrument.
        ufr: Ultimate forward rate as an annually compounded rate
            (decimal), e.g., 0.035 for 3.5%.
        convergence_point: Convergence point (in years) by which the forward
            rate is within the specified tolerance of the UFR.
        min_alpha: Minimum value of the alpha parameter used as a lower
            bound in calibration.
        convergence_tolerance_bp: Convergence tolerance in basis points (bp)
            used as tau in calibration. Default is 0.1 bp.
        max_maturity: Maximum maturity in whole years (inclusive) for which
            the outputs are returned. Default is 150 (year).

    Returns:
        Dict[str, np.ndarray]: A dictionary with arrays indexed by maturity (starting from 1):
            - "spot": Annual-compounded spot rates.
            - "forward": Annual-compounded one-year forward rates.
    """
    sw = SmithWilsonExtrapolator(
        instrument,
        coupon_frequency,
        rates,
        maturities,
        ufr,
        convergence_point,
        min_alpha,
        convergence_tolerance_bp
    )

    r = sw.extrapolate(max_maturity)

    return {
        "spot": r["zeroac"][1:max_maturity+1],
        "forward": r["forwardac"][1:max_maturity+1],
        "alpha": sw.alfa
    }
