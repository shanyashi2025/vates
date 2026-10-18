from dataclasses import dataclass
from enum import Enum
from typing import Self

from vates.alm.assets import Asset
from vates.alm._utils import Connector
from vates.alm.allocator._enums import AssetAllocationStrategy, AssetPurchaseMethod


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
        weight_range: (float | tuple[float, float, float]): Target weight for the allocation group; either a single
            float, or a three-element tuple representing (lower, target, upper).
        purchase_method (AssetPurchaseMethod): Method to purchase new asset for this allocation group.
        priority (int): The priority in which the allocation group will be processed, smaller integer represents
            higher priority.
    """
    strategy: AssetAllocationStrategy
    weight_range: tuple[float, float, float] | None = None
    purchase_method: AssetPurchaseMethod = AssetPurchaseMethod.NOT_USED
    priority: int = 999

    @classmethod
    def create(
        cls,
        strategy: AssetAllocationStrategy | str,
        weight_range: float | tuple[float, float, float] | None = None,
        purchase_method: AssetPurchaseMethod | str | None = None,
        priority: int = 999
    ) -> Self:
        if isinstance(strategy, str):
            strategy = AssetAllocationStrategy[strategy.upper()]
        if isinstance(weight_range, float):
            weight_range = (weight_range, weight_range, weight_range)
        if purchase_method is None:
            purchase_method = AssetPurchaseMethod.NOT_USED
        elif isinstance(purchase_method, str):
            purchase_method = AssetPurchaseMethod[purchase_method.upper()]
        return AssetAllocationParams(
            strategy=strategy,
            weight_range=weight_range,
            purchase_method=purchase_method,
            priority=priority,
        )

    def __post_init__(self):
        if not isinstance(self.strategy, AssetAllocationStrategy):
            raise TypeError(f"Invalid type of 'strategy': {type(self.strategy)}, expected 'AssetAllocationStrategy'.")

        if not isinstance(self.purchase_method, AssetPurchaseMethod):
            raise TypeError(f"Invalid type of 'purchase_method': {type(self.purchase_method)}, expected 'AssetPurchaseMethod'.")

        if self.weight_range is None:
            pass
        elif isinstance(self.weight_range, tuple):
            if len(self.weight_range) != 3:
                raise ValueError(f"Invalid len of 'weight_range': {len(self.weight_range)}, expected 3.")
            if not all(isinstance(x, float) for x in self.weight_range):
                raise TypeError(f"Invalid element type of 'weight_range' ({self.weight_range}), expected 'float'.")
            if not self.weight_range[0] <= self.weight_range[1] <= self.weight_range[2]:
                raise ValueError(f"Invalid 'weight_range' values: {self.weight_range}.")
        else:
            raise TypeError(f"Invalid type of 'weight_range': {type(self.weight_range)}, expected 'tuple'.")


class _AllocationGroup:
    """Asset allocation group.
    """

    __slots__ = ("name", "strategy", "weight_range", "purchase_method", "purchase_method", "priority",
                 "exist_conn", "profile_conn", "staged_conn", "size_basis", "size_range",
                 "profile_size", "_trade_orders", "_free_proceeds")

    def __init__(
        self,
        *,
        name: str,
        exist_assets: Connector[Asset] | list[Asset],
        strategy: AssetAllocationStrategy,
        weight_range: tuple[float, float, float] | None,
        purchase_method: AssetPurchaseMethod,
        priority: int,
        total_size: float,
        size_basis: str,
        profile_assets: Connector[Asset] | list[Asset] | None,
    ):
        self.name: str = name
        if isinstance(exist_assets, Connector):
            self.exist_conn: Connector[Asset] = exist_assets
        elif isinstance(exist_assets, list):
            self.exist_conn: Connector[Asset] = Connector(exist_assets)
        else:
            raise TypeError(f"Invalid type of 'exist_assets': {type(exist_assets)}, expected ('Connector', 'list').")
        self.strategy: AssetAllocationStrategy = strategy
        self.weight_range: tuple[float, float, float] | None = weight_range
        self.purchase_method: AssetPurchaseMethod = purchase_method
        self.priority: int = priority

        self.size_basis: str = size_basis
        if isinstance(profile_assets, Connector):
            self.profile_conn: Connector[Asset] = profile_assets
        elif isinstance(profile_assets, list):
            self.profile_conn: Connector[Asset] = Connector(profile_assets)
        elif profile_assets is None:
            self.profile_conn: Connector[Asset] = Connector()
        else:
            raise TypeError(f"Invalid type of 'profile_assets': {type(profile_assets)}, expected ('Connector', 'list').")

        self.size_range: tuple[float, float, float] | None = None
        if self.weight_range is not None:
            self.size_range = (total_size * self.weight_range[0],
                               total_size * self.weight_range[1],
                               total_size * self.weight_range[2])
        self.profile_size: float = self.profile_conn.sum_attr(self.size_basis) if self.profile_conn is not None else 0.0

        self.staged_conn: Connector[Asset] = Connector[Asset]()
        self._trade_orders: list[TradeOrder] = []
        self._free_proceeds: float = 0.0

    def take_out_free_proceeds(self) -> float:
        amount, self._free_proceeds = self._free_proceeds, 0.0
        return amount

    @property
    def current_size(self) -> float:
        return self.exist_size + self.staged_size

    @property
    def exist_size(self) -> float:
        return self.exist_conn.sum_attr(self.size_basis)

    @property
    def staged_size(self) -> float:
        return self.staged_conn.sum_attr(self.size_basis)

    @property
    def trade_orders(self) -> list[TradeOrder]:
        return self._trade_orders

    def append_trade_order(self, trade_order: TradeOrder) -> None:
        if not isinstance(trade_order, TradeOrder):
            raise TypeError(f"Invalid type of 'trade_order': '{type(trade_order)}', expected 'TradeOrder'.")
        self._trade_orders.append(trade_order)

    def clear_trade_orders(self) -> None:
        self._trade_orders = []

    @classmethod
    def merge_trade_orders(cls, trade_orders: list[TradeOrder]) -> list[TradeOrder]:
        if len(trade_orders) == 1:
            return trade_orders
        tmp_dict = {x: 0.0 for x in BuySellAction}
        for order in trade_orders:
            tmp_dict[order.action] += order.arg
        merged: list[TradeOrder] = []
        for act, arg in tmp_dict.items():
            if act != BuySellAction.NOOP and abs(arg) > 0.0:
                merged.append(TradeOrder(action=act, arg=arg))
        return merged

    def execute_trades(self, trade_orders: TradeOrder | list[TradeOrder] | None = None) -> None:
        if trade_orders is None:
            trade_orders = self._trade_orders
            consume_self_trade_orders = True
        else:
            consume_self_trade_orders = False
        trade_orders = trade_orders if isinstance(trade_orders, list) else [trade_orders]
        if len(trade_orders) == 0:
            return

        for order in self.merge_trade_orders(trade_orders):
            self.execute_trade(order)
        if consume_self_trade_orders:
            self._trade_orders = []  # reset after consumed

    def execute_trade(self, trade_order: TradeOrder) -> None:
        """Execute a trade.
        """
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

        self._free_proceeds += net_proceeds

    def is_allocation_satisfied(self, rel_tol: float = 1e-4) -> bool:
        """Validate if the allocation for a group is satisfied.

        Args:
            rel_tol (float): Tolerance relative to target size for validation (default 0.0001).

        Returns:
            bool: True if the allocation is satisfied, False otherwise.
        """
        if self.strategy in (AssetAllocationStrategy.UNCONSTRAINED, AssetAllocationStrategy.NO_TRADE):
            return True

        lower, target, upper = self.size_range

        if self.strategy == AssetAllocationStrategy.BUY_HOLD:
            return lower - rel_tol * target <= self.current_size
        elif self.strategy == AssetAllocationStrategy.BUY_SELL:
            return lower - rel_tol * target <= self.current_size <= upper + rel_tol * target
        else:
            raise ValueError(f"{self.name}: invalid asset allocation strategy {self.strategy}.")
