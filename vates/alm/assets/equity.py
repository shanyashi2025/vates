import pandas as pd
import warnings

from vates._core import TDepVariable
from vates.utils import check_calc_time
from vates.alm.enums import AssetClassification
from vates.alm.econs import Currency, EquityIndex
from vates.alm.assets.asset_base import Asset


class Equity(Asset):
    """
    Represents an equity asset in the portfolio.

    Attributes:
        _equity_index (EquityIndex): Associated equity index.
        _mv (float): Market value of the equity asset.
        _fav (float): Fund accouting value of the equity asset.
        tdv_dividend (float): Dividend for the current period.
    """
    __slots__ = ('_equity_index', '_mv', '_fav', '_cash_flow',
                 'tdv_cash_flow', 'tdv_dividend', 'tdv_mv_bd', 'tdv_mv_ad', 'tdv_fav_bd', 'tdv_fav_ad',)

    def __init__(self, model, asset_id: str, is_profile: bool, currency: Currency | None, asset_category: str, fund_id: str,
                 allocation_group: str, mv: float, fav: float, equity_index: EquityIndex,
                 classification: AssetClassification, purchase_date: pd.Period | None=None):
        """
        Initialize an Equity asset.

        Args:
            asset_id (str): Asset identifier.
            is_profile (bool): Ture if profile asset, False if existing asset.
            currency (Currency): Asset currency.
            asset_category (str): Asset category.
            fund_id (str): Fund identifier.
            allocation_group (str): Allocation group.
            mv (float): Market value.
            fav (float): Fund accouting value.
            equity_index (EquityIndex): Associated equity index.
            classification (AssetClassification): Asset classification.
            purchase_date (pd.Period | None): Purchase date, default to initilization date.
        """
        super().__init__(model, asset_id, is_profile, 1, purchase_date, currency, classification, asset_category,
                         fund_id, allocation_group)
        t = self.time
        self._equity_index: EquityIndex = equity_index
        self._mv: float = mv
        self._fav: float = fav
        if self.classification == AssetClassification.FVTPL:
            if abs(self._mv - self._fav) > 1e-8:
                self._fav = self._mv
                warnings.warn(f'Equity asset {self.asset_id}: fav is forcedly set to mv for FVTPL.')
        elif self.classification == AssetClassification.FVOCI:
            pass
        else:
            raise ValueError(f'Equity asset {self.asset_id}: invalid asset classification: {self.classification}, '
                             f'epxected FVTPL or FVOCI')

        if abs(self._mv) < 1e-8: self._mv = 1e-8  # to prevent crash when proportionally buy new asset
        if abs(self._fav) < 1e-8: self._fav = 1e-8

        self._cash_flow: float = 0.0

        self.tdv_cash_flow: TDepVariable = TDepVariable(model, "cash_flow", asset_id, 'equity')
        self.tdv_dividend: TDepVariable = TDepVariable(model, "dividend", asset_id, 'equity')
        self.tdv_mv_bd: TDepVariable = TDepVariable(model, "mv_bd", asset_id, 'equity')
        self.tdv_mv_ad: TDepVariable = TDepVariable(model, "mv_ad", asset_id, 'equity')
        self.tdv_fav_bd: TDepVariable = TDepVariable(model, "fav_bd", asset_id, 'equity')
        self.tdv_fav_ad: TDepVariable = TDepVariable(model, "fav_ad", asset_id, 'equity')

        if not is_profile:
            self.tdv_mv_ad[t] = self._mv
            self.tdv_fav_ad[t] = self._fav

    @property
    def is_alive(self) -> bool:
        return True

    @check_calc_time({"roll_forward": -1, "complete_dealing": -1}, "roll_forward")
    def roll_forward(self, **kwargs) -> None:
        """
        Roll the equity asset forward one period, updating value and dividend.
        """
        t = self.time

        if self._equity_index.last_update != t:
            raise ValueError(f"{self._equity_index.index_id} is not updated on {t} ({self.period}).")

        dividend = self._mv * self._equity_index.dividend_yield
        self._cash_flow = dividend
        self._mv = self._mv * (1 + self._equity_index.capital_growth)  # total return = capital growth + dividend yield
        if self.classification == AssetClassification.FVTPL:
            self._fav = self._mv
        elif self.classification == AssetClassification.FVOCI:
            pass  # fav is kept unchanged, as capital growth not recognised

        self.tdv_dividend[t] = dividend
        self.tdv_cash_flow[t] = self._cash_flow
        self.tdv_mv_bd[t] = self._mv
        self.tdv_fav_bd[t] = self._fav

    def buy_propn(self, propn: float) -> None:
        """
        Buy a proportion of the equity asset.

        Args:
            propn (float): Proportion to buy.

        Raises:
            ValueError: If propn is negative.
        """
        if propn < 0:
            raise ValueError("Can not buy negative proportion of an exsiting equity.")
        if self.classification != AssetClassification.FVTPL:
            raise ValueError(f"Equity asset {self.asset_id}: can not buy proportion of a non-FVTPL equitiy.")
        self._mv += self._mv * propn
        self._fav += self._fav * propn

    def sell_propn(self, propn: float) -> None:
        """
        Sell a proportion of the equity asset.

        Args:
            propn (float): Proportion to sell.

        Raises:
            ValueError: If propn is negative.
        """
        if propn < 0: raise ValueError("Can not sell negative proportion of an exsiting equity.")
        if propn > 1: raise ValueError("Can not sell >100% proportion of an exsiting equity.")
        self._mv -= self._mv * propn
        self._fav -= self._fav * propn

    def buy_profile_scale(self, scale: float, list_to_append: list | None=None) -> None:
        """
        Scale the equity profile by a factor.

        Args:
            scale (float): Scaling factor.
            list_to_append (list): Append to list.
        """
        if not self._is_profile: raise ValueError("This equity object is not a profile.")
        if scale < 0: raise ValueError("Can not scale equity profile by a negative number.")
        self._mv = self._mv * scale
        self._fav = self._fav * scale
        self._is_profile = False
        if list_to_append is not None: list_to_append.append(self)

    @check_calc_time({"complete_dealing": -1, "roll_forward": 0}, "complete_dealing")
    def complete_dealing(self, **kwargs) -> None:
        """
        Update the equity asset after dealing.
        """
        t = self.time
        self.tdv_mv_ad[t] = self._mv
        self.tdv_fav_ad[t] = self._fav

    @property
    def mv(self) -> float:
        """float: Market value of the equity asset."""
        return self._mv

    @property
    def fav(self) -> float:
        """float: Fund accouting value of the equity asset."""
        return self._fav

    @property
    def bsv(self) -> float:
        """float: Balance sheet value of the equity asset."""
        if self.classification in (AssetClassification.FVTPL, AssetClassification.FVOCI):
            return self._mv
        else:
            raise ValueError(f'Equity asset {self.asset_id}: invalid asset classification: {self.classification}, '
                             f'epxected FVTPL or FVOCI')

    @property
    def cash_flow(self) -> float:
        """float: Cash flow in period"""
        return self._cash_flow

    @property
    def arr_cash_flow(self) -> TDepVariable:
        """TDepVariable: Cash flow array"""
        return self.tdv_cash_flow
