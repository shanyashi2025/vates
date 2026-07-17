import numpy as np
import numpy.typing as npt
import math
from typing import Callable


def interpolate_interest_rates(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], *, method: str, **kwargs
                               ) -> npt.NDArray[np.float64]:
    """
    Perform interpolation using the specified kind.

    Args:
        x: Array of x-coordinates (must be sorted in ascending order)
        y: Array of y-coordinates corresponding to x
        method: Interpolation method to use

    Returns:
        npt.NDArray[np.float64]: Array of interpolated values.

    Raises:
        ValueError: If the specified method is not supported.
    """
    method = method.lower()
    func = InterestRateInterpolator.get_func(method.lower())
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
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[0] = y[0]

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
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[0] = y[0]

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
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[0] = y[0]

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
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[0] = y[0]

        for i in range(1, len(x)):
            x0, x1 = int(x[i - 1]), int(x[i])
            y0, y1 = y[i - 1], y[i]

            for j in range(x0 + 1, x1 + 1):
                y_interp[j] = y1

        return y_interp

    @classmethod
    def exponential(cls, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], *, time_interval: float,
                    forward_rate_cc: float = math.log(1.045)) -> npt.NDArray[np.float64]:
        """
        Perform exponential interpolation between data points of discount factor or zero coupon bond (zcb).

        Args:
            x: Array of x-coordinates (must be sorted in ascending order)
            y: Array of y-coordinates corresponding to x (must be positive)
            time_interval: Time interval in years (e.g. 1 means x repsents year, 1/12 means x repsents months)
            forward_rate_cc: Constant forward rate (continuously compounded)

        Returns:
            npt.NDArray[np.float64]: Array of interpolated y-values for all integer x-values from 0 to max(x).
        """
        y_interp = np.zeros(int(x[-1]) + 1)
        y_interp[0] = y[0]
        z = np.zeros(int(x[-1]) + 1)
        z[:] = np.exp(-forward_rate_cc * np.arange(0, int(x[-1]) + 1) * time_interval)  # exp(-ft)

        for i in range(1, len(x)):
            x0, x1 = int(x[i - 1]), int(x[i])
            y0, y1 = y[i - 1], y[i]

            for j in range(x0 + 1, x1 + 1):
                w0 = (z[j - x0] - z[x1 - x0]) / (1 - z[x1 - x0])
                w1 = (1 - z[j - x0]) / (1 - z[x1 - x0])
                y_interp[j] = y0 * w0 + y1 * w1

        return y_interp

    @classmethod
    def get_func(cls, method: str) -> Callable:
        if method == "linear":
            return cls.linear
        elif method in ("loglinear", "log_linear"):
            return cls.loglinear
        elif method == "geometric":
            return cls.geometric
        elif method in ("next", "next_point", "next_value"):
            return cls.next
        elif method in ("zcb_exponential", "exponential"):
            return cls.exponential
        else:
            raise ValueError(f"Invalid interpolation method: {method}")
