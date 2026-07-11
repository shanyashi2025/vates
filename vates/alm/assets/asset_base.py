from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import pandas as pd

from vates._core import ProjModelEngine
from vates.alm.enums import AssetRepBasis, AssetClassification
from vates.alm.econs import Currency

class Asset(ABC):
    """
    Abstract base class for all financial assets.

    Attributes:
        _asset_id (str): Unique identifier for the asset.
        _is_profile (bool): Ture if profile asset, False if existing asset.
        _units (float): Number of assets
        _purchase_date (pd.Period): Purchase date.
        _currency (Currency): Currency of the asset.
        _classification (AssetClassification): Asset classification.
        _asset_category (str): Asset category.
        _fund_id (str): Associated fund identifier.
        _allocation_group (str): Allocation group for the asset.
    """
    __slots__ = ('__dict__', '__weakref__', 'time', 'period', '_asset_id', '_is_profile', '_units', '_purchase_date',
                 '_currency', '_classification', '_asset_category', '_fund_id', '_allocation_group', '_tc_dict')

    def __init__(
        self,
        model_engine: ProjModelEngine,
        *,
        asset_id: str,
        is_profile: bool,
        units: float,
        purchase_date: pd.Period | None,
        currency: Currency | None,
        classification: AssetClassification,
        asset_category: str,
        fund_id: str,
        allocation_group: str
    ):
        """
        Initialize the Asset.

        Args:
            model_engine: Model engine object.
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            units (float): Number of assets
            purchase_date (pd.Period): Purchase date. Set to initilization date if input is None.
            currency (Currency): Asset currency.
            classification (AssetClassification): Asset classification.
            asset_category (str): Asset category.
            fund_id (str): Fund identifier.
            allocation_group (str): Allocation group.
        """
        model_engine.attach_time_observer(self)
        self.time: int = model_engine.time
        self.period: pd.Period = model_engine.period

        self._asset_id: str = asset_id
        self._is_profile: bool = is_profile
        self._units: float = units
        self._purchase_date: pd.Period = self.period if purchase_date is None else purchase_date
        self._currency: Currency | None = currency
        self._classification: AssetClassification = classification
        self._asset_category: str = asset_category
        self._fund_id: str = fund_id
        self._allocation_group: str = allocation_group
        self._tc_dict: dict[str, int] = {"roll_forward": self.time}
        if not self._is_profile:
            self._tc_dict['dealing'] = self.time

    def sync_time(self, subject: ProjModelEngine) -> None:
        self.time = subject.time
        self.period = subject.period

    @property
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def is_profile(self) -> bool:
        return self._is_profile

    @property
    def units(self) -> float:
        return self._units

    @property
    def currency(self) -> Currency:
        return self._currency

    @property
    def asset_category(self) -> str:
        return self._asset_category

    @property
    def fund_id(self) -> str:
        return self._fund_id

    @property
    def allocation_group(self) -> str:
        return self._allocation_group

    @property
    def classification(self) -> AssetClassification:
        return self._classification

    @property
    def purchase_date(self) -> pd.Period:
        return self._purchase_date

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """
        Abstract property for whether asset is alive.

        Returns:
            bool: True if asset is alive, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def mv(self):
        """
        Abstract property for market value.

        Returns:
            float: Market value (to be implemented by subclasses).
        """
        pass

    @property
    @abstractmethod
    def fav(self):
        """
        Abstract property for fund accounting value.

        Returns:
            float: Fund accounting value (to be implemented by subclasses).
        """
        pass

    @property
    @abstractmethod
    def bsv(self) :
        """
        Abstract property for balance sheet value.

        Returns:
            float: Balance sheet value (to be implemented by subclasses).
        """
        pass

    @property
    def rep_value(self) -> npt.NDArray[np.float64]:
        """
        Get all reported values as a numpy array.

        Returns:
            npt.NDArray[np.float64]: Array of reported values.
        """
        result = np.zeros(len(AssetRepBasis))

        result[AssetRepBasis.MV.value] = self.mv
        result[AssetRepBasis.FAV.value] = self.fav
        result[AssetRepBasis.BSV.value] = self.bsv

        return result

    @property
    def last_roll_forward(self) -> int:
        """int: Last roll forward time index."""
        return self._tc_dict.get('roll_forward', None)

    @property
    def last_dealing(self) -> int:
        """int: Last update after dealing time index."""
        return self._tc_dict.get('dealing', None)

    @abstractmethod
    def roll_forward(self, *args, **kwargs):
        """
        Abstract method to roll the asset forward in time.
        """
        pass

    @property
    @abstractmethod
    def cash_flow(self) -> float:
        """
        Abstract method to get the cash flow for the asset in period.
        """
        pass

    @property
    @abstractmethod
    def arr_cash_flow(self):
        """
        Abstract method to get the cash flow for the asset.
        """
        pass

    @abstractmethod
    def buy_propn(self, *args, **kwargs):
        """
        Abstract method to buy a proportion of the asset.
        """
        pass

    @abstractmethod
    def sell_propn(self, *args, **kwargs):
        """
        Abstract method to sell a proportion of the asset.
        """
        pass

    @abstractmethod
    def buy_profile_scale(self, *args, **kwargs):
        """
        Abstract method to scale the asset profile by a factor.
        """
        pass

    @abstractmethod
    def close_dealing(self, *args, **kwargs):
        """
        Abstract method to update the asset after dealing.
        """
        pass

    def __str__(self) -> str:
        return self.asset_id
