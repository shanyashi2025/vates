from dataclasses import dataclass
from enum import Enum
from typing import Self

from vates.alm.assets import Asset
from vates.alm._utils import Connector
from vates.alm.allocation._enums import AssetAllocationStrategy, AssetPurchaseMethod


class BuySellAction(Enum):
    """Enum for asset buy/sell action. Must correspond to methods of asset classes"""
    NOOP = "NOOP"
    BUY_EXIST = "buy_propn"
    BUY_PROFILE = "buy_profile"
    SELL = "sell_propn"


@dataclass(slots=True, frozen=True)
class TradeOrder:
    action: BuySellAction
    arg: float = 0.0


@dataclass(slots=True, frozen=True)
class AssetAllocationParams:
    """Allocation parameters for an allocation group.

    Attributes:

        strategy (AssetAllocationStrategy): Allocation strategy for this allocation group.
        target_weight: (float | tuple[float, float, float]): Target weight for the allocation group; either a single
            float, or a three-element tuple representing (floor, target, cap).
        purchase_method (AssetPurchaseMethod): Method to purchase new asset for this allocation group.
        sequence (int): The sequence in which the allocation group will be processed, must be unique and consecutive
            integer starting from 1.
    """
    strategy: AssetAllocationStrategy
    target_weight: tuple[float, float, float] | None = None
    purchase_method: AssetPurchaseMethod = AssetPurchaseMethod.NOT_USED
    sequence: int = 999

    @classmethod
    def create(cls,
        strategy: AssetAllocationStrategy | str,
        target_weight: float | tuple[float, float, float] | None = None,
        purchase_method: AssetPurchaseMethod | str | None = None,
        sequence: int = 999
    ) -> Self:
        if isinstance(strategy, str):
            strategy = AssetAllocationStrategy[strategy.upper()]
        if isinstance(target_weight, float):
            target_weight = (target_weight, target_weight, target_weight)
        if purchase_method is None:
            purchase_method = AssetPurchaseMethod.NOT_USED
        elif isinstance(purchase_method, str):
            purchase_method = AssetPurchaseMethod[purchase_method.upper()]
        return AssetAllocationParams(
            strategy=strategy,
            target_weight=target_weight,
            purchase_method=purchase_method,
            sequence=sequence,
        )

    def __post_init__(self):
        if not isinstance(self.strategy, AssetAllocationStrategy):
            raise TypeError(f"Invalid type of 'strategy': {type(self.strategy)}, expected 'AssetAllocationStrategy'.")

        if not isinstance(self.purchase_method, AssetPurchaseMethod):
            raise TypeError(f"Invalid type of 'purchase_method': {type(self.purchase_method)}, expected 'AssetPurchaseMethod'.")

        if self.target_weight is None:
            pass
        elif isinstance(self.target_weight, tuple):
            if len(self.target_weight) != 3:
                raise ValueError(f"Invalid len of 'target_weight': {len(self.target_weight)}, expected 3.")
            if not all(isinstance(x, float) for x in self.target_weight):
                raise TypeError(f"Invalid element type of 'target_weight' ({self.target_weight}), expected 'float'.")
            if not self.target_weight[0] <= self.target_weight[1] <= self.target_weight[2]:
                raise ValueError(f"Invalid 'target_weight' values: {self.target_weight}.")
        else:
            raise TypeError(f"Invalid type of 'target_weight': {type(self.target_weight)}, expected 'tuple'.")


class AssetAllocationGroup:
    """Asset allocation group.

    Attributes:
        _name (str): Name of the allocation group.
        _params (AssetAllocationParams): Static allocation parameters:
            - strategy: AssetAllocationStrategy
            - target_weight: tuple[float, float, float]
            - purchase_method: AssetPurchaseMethod | None
            - sequence: int
    """

    __slots__ = ("_name", "_params", "_exist", "_profile", "_staged", "_size_basis", "_target_size",
                 "_profile_size", "_trade_decision", "_net_proceeds")

    def __init__(
        self,
        *,
        name: str,
        exist_assets: Connector[Asset] | list[Asset],
        params: AssetAllocationParams,
        total_size: float,
        size_basis: str,
        profile_assets: Connector[Asset] | list[Asset] | None,
    ):
        self._name: str = name
        if isinstance(exist_assets, Connector):
            self._exist: Connector[Asset] = exist_assets
        elif isinstance(exist_assets, list):
            self._exist: Connector[Asset] = Connector(exist_assets)
        else:
            raise TypeError(f"Invalid type of 'exist_assets': {type(exist_assets)}, expected ('Connector', 'list').")
        self._params: AssetAllocationParams = params
        self._size_basis: str = size_basis
        if isinstance(profile_assets, Connector):
            self._profile: Connector[Asset] = profile_assets
        elif isinstance(profile_assets, list):
            self._profile: Connector[Asset] = Connector(profile_assets)
        elif profile_assets is None:
            self._profile: Connector[Asset] = Connector()
        else:
            raise TypeError(f"Invalid type of 'profile_assets': {type(profile_assets)}, expected ('Connector', 'list').")

        self._target_size: tuple[float, float, float] | None = None
        if self._params.target_weight is not None:
            self._target_size = (total_size * self._params.target_weight[0],
                                 total_size * self._params.target_weight[1],
                                 total_size * self._params.target_weight[2])
        self._profile_size: float = self._profile.sum_attr(self._size_basis) if self._profile is not None else 0.0

        self._staged: Connector[Asset] = Connector[Asset]()
        self._trade_decision: TradeOrder = TradeOrder(action=BuySellAction.NOOP)
        self._net_proceeds: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def sequence(self) -> int:
        return self._params.sequence

    @property
    def strategy(self) -> AssetAllocationStrategy:
        return self._params.strategy

    @property
    def purchase_method(self) -> AssetPurchaseMethod:
        return self._params.purchase_method

    @property
    def size_basis(self) -> str:
        return self._size_basis

    @property
    def exist_conn(self) -> Connector[Asset]:
        return self._exist

    @property
    def profile_conn(self) -> Connector[Asset]:
        return self._profile

    @property
    def staged_conn(self) -> Connector[Asset]:
        return self._staged

    def take_out_net_proceeds(self) -> float:
        amount, self._net_proceeds = self._net_proceeds, 0.0
        return amount

    @property
    def current_size(self) -> float:
        return self.exist_size + self.staged_size

    @property
    def exist_size(self) -> float:
        return self._exist.sum_attr(self.size_basis)

    @property
    def staged_size(self) -> float:
        return self._staged.sum_attr(self.size_basis)

    @property
    def profile_size(self) -> float:
        return self._profile_size

    @property
    def target_weight(self) -> tuple[float, float, float] | None:
        return self._params.target_weight

    @property
    def target_size(self) -> tuple[float, float, float] | None:
        return self._target_size

    @property
    def trade_decision(self) -> TradeOrder:
        return self._trade_decision

    def make_trade_decision(self) -> None:
        """Make the trade decision for an allocation group.

        Raises:
            ValueError: If an invalid asset buy/sell approach is provided.
        """
        strategy = self._params.strategy
        purchase_method = self._params.purchase_method
        if strategy == AssetAllocationStrategy.NO_TRADE:
            self._trade_decision = TradeOrder(action=BuySellAction.NOOP)
        elif strategy == AssetAllocationStrategy.BUY_HOLD:
            if self.current_size < self._target_size[0]:
                if purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    self._trade_decision = TradeOrder(
                        action=BuySellAction.BUY_EXIST,
                        arg=(self._target_size[1] - self.current_size) / self.current_size)
                elif purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    self._trade_decision = TradeOrder(
                        action=BuySellAction.BUY_PROFILE,
                        arg=(self._target_size[1] - self.current_size) / self.profile_size)
                else:
                    raise ValueError(f"{self.name}: invalid purchase method {purchase_method} for 'BUY_HOLD'.")
            else:
                self._trade_decision = TradeOrder(action=BuySellAction.NOOP)
        elif strategy == AssetAllocationStrategy.BUY_SELL:
            if self.current_size < self._target_size[0]:
                if purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING:
                    self._trade_decision = TradeOrder(
                        action=BuySellAction.BUY_EXIST,
                        arg=(self._target_size[1] - self.current_size) / self.current_size)
                elif purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE:
                    self._trade_decision = TradeOrder(
                        action=BuySellAction.BUY_PROFILE,
                        arg=(self._target_size[1] - self.current_size) / self.profile_size)
                else:
                    raise ValueError(f"{self.name}: invalid purchase method {purchase_method} for 'BUY_SELL'.")
            elif self.current_size > self._target_size[2]:
                self._trade_decision = TradeOrder(
                    action=BuySellAction.SELL,
                    arg=(self.current_size - self._target_size[1]) / self.current_size)
            else:
                self._trade_decision = TradeOrder(action=BuySellAction.NOOP)
        else:
            raise ValueError(f"{self.name}: invalid allocation strategy: {strategy}")

    def execute_trade(self, trade_order: TradeOrder | None = None) -> None:
        """Execute a trade.
        """
        if trade_order is None:
            trade_order = self._trade_decision
            consume_self_trade_decision = True
        else:
            consume_self_trade_decision = False

        net_proceeds: float = 0.0  # proceeds received from disposal (sell), minus proceeds spent to purchase (buy)
        action, arg = trade_order.action, trade_order.arg

        if action == BuySellAction.NOOP:
            pass
        elif action == BuySellAction.SELL:
            for asset in self.exist_conn:
                mv_bd = asset.market_value
                func = getattr(asset, action.value)
                func(propn=arg)
                mv_ad = asset.market_value
                net_proceeds += mv_bd - mv_ad
        elif action == BuySellAction.BUY_EXIST:
            for asset in self.exist_conn:
                mv_bd = asset.market_value
                func = getattr(asset, action.value)
                func(propn=arg)
                mv_ad = asset.market_value
                net_proceeds += mv_bd - mv_ad
        elif action == BuySellAction.BUY_PROFILE:
            if self.profile_conn is None:
                raise ValueError("Can't buy assets from empty profile.")
            for profile_asset in self.profile_conn:
                func = getattr(profile_asset, action.value)
                new_asset = func(scale=arg)
                self.staged_conn.append(new_asset) # append to asset list of the staged
                net_proceeds -= new_asset.market_value

        self._net_proceeds += net_proceeds

        if consume_self_trade_decision:
            self._trade_decision = TradeOrder(action=BuySellAction.NOOP)  # reset after consumed

    def is_allocation_satisfied(self, rel_tol: float = 1e-4) -> bool:
        """Validate if the allocation for a group is satisfied.

        Args:
            rel_tol (float): Tolerance relative to target size for validation (default 0.0001).

        Returns:
            bool: True if the allocation is satisfied, False otherwise.
        """
        if self.strategy in (AssetAllocationStrategy.RESIDUAL, AssetAllocationStrategy.NO_TRADE):
            return True

        flr, tgt, cap = self._target_size

        if self.strategy == AssetAllocationStrategy.BUY_HOLD:
            return flr - rel_tol * tgt <= self.current_size
        elif self.strategy == AssetAllocationStrategy.BUY_SELL:
            return flr - rel_tol * tgt <= self.current_size <= cap + rel_tol * tgt
        else:
            raise ValueError(f"{self.name}: invalid asset allocation strategy {self.strategy}.")
