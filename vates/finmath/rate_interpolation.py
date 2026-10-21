import numpy as np
import numpy.typing as npt
import math
from typing import Callable, Literal


def interpolate_interest_rates(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], *, interp_method: str, **kwargs
                               ) -> npt.NDArray[np.float64]:
    """
    Perform interpolation using the specified kind.

    Args:
        x: Array of x-coordinates (must be sorted in ascending order)
        y: Array of y-coordinates corresponding to x
        interp_method: Interpolation method to use

    Returns:
        npt.NDArray[np.float64]: Array of interpolated y-values for all integer x-values from 0 to max(x).
            Values before `x[0]` cannot be interpolated and are set to 0.0.

    Raises:
        ValueError: If the specified method is not supported.
    """
    interp_method = interp_method.lower()
    func = InterestRateInterpolator.get_func(interp_method.lower())
    return func(x, y, **kwargs)


class InterestRateInterpolator:

    @classmethod
    def linear(cls, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Perform linear interpolation between data points.

        Args:
            x: Array of x-coordinates (must be sorted in ascending order)
            y: Array of y-coordinates corresponding to x

        Returns:
            npt.NDArray[np.float64]: Array of interpolated y-values for all integer x-values from 0 to max(x).
            Values before `x[0]` cannot be interpolated and are set to 0.0.
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[int(x[0])] = y[0]

        for i in range(1, len(x)):
            x0, x1 = int(x[i - 1]), int(x[i])
            y0, y1 = y[i - 1], y[i]
            k = (y1 - y0) / (x1 - x0)

            for j in range(x0 + 1, x1 + 1):
                y_interp[j] = y0 + k * (j - x0)

        return y_interp

    @classmethod
    def loglinear(cls, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Perform log-linear interpolation between data points.

        This method interpolates in log-space, which is useful for rates and discount factors that should remain positive.

        Args:
            x: Array of x-coordinates (must be sorted in ascending order)
            y: Array of y-coordinates corresponding to x (must be positive)

        Returns:
            npt.NDArray[np.float64]: Array of interpolated y-values for all integer x-values from 0 to max(x).
            Values before `x[0]` cannot be interpolated and are set to 0.0.
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[int(x[0])] = y[0]

        for i in range(1, len(x)):
            x0, x1 = int(x[i - 1]), int(x[i])
            log_y0, log_y1 = math.log(y[i - 1]), math.log(y[i])
            k = (log_y1 - log_y0) / (x1 - x0)

            for j in range(x0 + 1, x1 + 1):
                y_interp[j] = math.exp(log_y0 + k * (j - x0))

        return y_interp

    @classmethod
    def geometric(cls, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Perform geometric interpolation between data points.

        This method assumes exponential growth/decay between points.

        Args:
            x: Array of x-coordinates (must be sorted in ascending order)
            y: Array of y-coordinates corresponding to x (must be positive)

        Returns:
            npt.NDArray[np.float64]: Array of interpolated y-values for all integer x-values from 0 to max(x).
            Values before `x[0]` cannot be interpolated and are set to 0.0.
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[int(x[0])] = y[0]

        for i in range(1, len(x)):
            x0, x1 = int(x[i - 1]), int(x[i])
            y0, y1 = y[i - 1], y[i]
            k = (y1 / y0) ** (1 / (x1 - x0))

            for j in range(x0 + 1, x1 + 1):
                y_interp[j] = y0 * (k ** (j - x0))

        return y_interp

    @classmethod
    def next(cls, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Take value of next data point.

        Args:
            x: Array of x-coordinates (must be sorted in ascending order)
            y: Array of y-coordinates corresponding to x (must be positive)

        Returns:
            npt.NDArray[np.float64]: Array of interpolated y-values for all integer x-values from 0 to max(x).
            Values before `x[0]` cannot be interpolated and are set to 0.0.
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[int(x[0])] = y[0]

        for i in range(1, len(x)):
            x0, x1 = int(x[i - 1]), int(x[i])
            y0, y1 = y[i - 1], y[i]

            for j in range(x0 + 1, x1 + 1):
                y_interp[j] = y1

        return y_interp

    @classmethod
    def exponential(cls, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], *,
                    interval_unit: Literal["Y", "M"] = "Y", forward_rate_cc: float = math.log(1.045)
                    ) -> npt.NDArray[np.float64]:
        """
        Perform exponential interpolation between data points of discount factor or zero coupon bond (zcb).

        Args:
            x: Array of x-coordinates (must be sorted in ascending order)
            y: Array of y-coordinates corresponding to x (must be positive)
            interval_unit (Literal["Y", "M"]): Grid interval unit, year or month. Defaults to year.
            forward_rate_cc: Constant forward rate (continuously compounded)

        Returns:
            npt.NDArray[np.float64]: Array of interpolated y-values for all integer x-values from 0 to max(x).
            Values before `x[0]` cannot be interpolated and are set to 0.0.

        Raises:
            ValueError: If `interval_unit` is not `"Y"` or `"M"`.
        """
        if interval_unit == "Y":
            interval_in_years = 1
        elif interval_unit == "M":
            interval_in_years = 1 / 12
        else:
            raise ValueError(f"Invalid {interval_unit=}, expected 'Y' or 'M'.")

        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[int(x[0])] = y[0]
        z = np.exp(-forward_rate_cc * np.arange(0, int(x[-1]) + 1) * interval_in_years)  # exp(-ft)

        for i in range(1, len(x)):
            x0, x1 = int(x[i - 1]), int(x[i])
            y0, y1 = y[i - 1], y[i]

            for j in range(x0 + 1, x1 + 1):
                w0 = (z[j - x0] - z[x1 - x0]) / (1 - z[x1 - x0])
                w1 = (1 - z[j - x0]) / (1 - z[x1 - x0])
                y_interp[j] = y0 * w0 + y1 * w1

        return y_interp

    @classmethod
    def cubic_spline(cls, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], *,
                     bc_type: str = "not-a-knot") -> npt.NDArray[np.float64]:
        """
        Perform cubic-spline interpolation between data points.

        Requires the optional dependency `scipy`. Values are not shape-preserving: the spline may
        overshoot or go negative between data points.

        Args:
            x: Array of x-coordinates (must be sorted in ascending order)
            y: Array of y-coordinates corresponding to x
            bc_type: Boundary condition passed to `scipy.interpolate.CubicSpline`.

        Returns:
            npt.NDArray[np.float64]: Array of interpolated y-values for all integer x-values from 0 to max(x).
            Values before `x[0]` cannot be interpolated and are set to 0.0.

        Raises:
            ImportError: If `scipy` is not installed.
        """
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            raise ImportError("Need to install 'scipy' library (`pip install scipy`).") from None
        cs = CubicSpline(x, y, bc_type=bc_type)
        xs = np.arange(int(x[0]), int(x[-1]) + 1)
        return np.concatenate((np.zeros(int(x[0])), cs(xs)))

    @classmethod
    def get_func(cls, interp_method: str, /) -> Callable:
        if interp_method == "linear":
            return cls.linear
        elif interp_method in ("loglinear", "log_linear"):
            return cls.loglinear
        elif interp_method == "geometric":
            return cls.geometric
        elif interp_method in ("next", "next_point", "next_value"):
            return cls.next
        elif interp_method in ("zcb_exponential", "exponential"):
            return cls.exponential
        elif interp_method in ("cubic_spline", "cubic"):
            return cls.cubic_spline
        else:
            raise ValueError(f"Invalid interpolation method: {interp_method}")
