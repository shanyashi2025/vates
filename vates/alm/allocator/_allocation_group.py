import warnings
from dataclasses import dataclass
from enum import Enum

from vates.alm.assets import Asset
from vates.alm._utils import Connector
from vates.alm.allocator._enums import AssetAllocationStrategy, AssetPurchaseMethod, AssetRebalanceObjective


class BuySellAction(Enum):
    """Enum for asset buy/sell action. Must correspond to methods of asset classes"""
    EXCEPTION = "EXCEPTION"
    PASS = "PASS"
    BUY_EXIST = "buy_propn"
    BUY_PROFILE = "buy_profile"
    SELL = "sell_propn"


@dataclass(slots=True, frozen=True)
class TradeOrder:
    action: BuySellAction
    arg: float = 0.0


@dataclass(slots=True, frozen=True)
class PrimaryRebalanceObjective:
    below_range: AssetRebalanceObjective
    within_range: AssetRebalanceObjective
    above_range: AssetRebalanceObjective

    @classmethod
    def create(cls, below_range: AssetRebalanceObjective | str | None, within_range: AssetRebalanceObjective | str | None,
               above_range: AssetRebalanceObjective | str | None) -> 'PrimaryRebalanceObjective':
        if below_range is None:
            below_range = AssetRebalanceObjective.NOT_USED
        elif isinstance(below_range, str):
            below_range = AssetRebalanceObjective[below_range.upper()]
        if within_range is None:
            within_range = AssetRebalanceObjective.NOT_USED
        elif isinstance(within_range, str):
            within_range = AssetRebalanceObjective[within_range.upper()]
        if above_range is None:
            above_range = AssetRebalanceObjective.NOT_USED
        elif isinstance(above_range, str):
            above_range = AssetRebalanceObjective[above_range.upper()]
        return PrimaryRebalanceObjective(below_range=below_range, within_range=within_range, above_range=above_range)

    def __post_init__(self):
        if not isinstance(self.below_range, AssetRebalanceObjective):
            raise TypeError(f"Invalid type of 'below_range': {type(self.below_range)}, expected 'AssetTradeDirection'")
        if not isinstance(self.within_range, AssetRebalanceObjective):
            raise TypeError(f"Invalid type of 'within_range': {type(self.within_range)}, expected 'AssetTradeDirection'")
        if not isinstance(self.above_range, AssetRebalanceObjective):
            raise TypeError(f"Invalid type of 'above_range': {type(self.above_range)}, expected 'AssetTradeDirection'")


@dataclass(slots=True, frozen=True)
class AssetAllocationParams:
    """Allocation parameters for an allocation group.

    Attributes:

        strategy (AssetAllocationStrategy): Allocation strategy for this allocation group.
        weight_box (tuple[float, float, float]): Target weight for the allocation group,
            a three-element tuple representing (lower, target, upper).
        primary_objective (PrimaryRebalanceObjective): Primary rebalance objective.
        purchase_method (AssetPurchaseMethod): Method to purchase new asset for this allocation group.
        priority (int): The priority in which the allocation group will be processed, smaller integer represents
            higher priority.
    """
    strategy: AssetAllocationStrategy
    weight_box: tuple[float, float, float] | None
    primary_objective: PrimaryRebalanceObjective
    purchase_method: AssetPurchaseMethod
    priority: int

    @classmethod
    def create(
        cls,
        strategy: AssetAllocationStrategy | str,
        weight_box: tuple[float, float, float] | None = None,
        primary_objective: dict[str, str | AssetRebalanceObjective | None] | PrimaryRebalanceObjective | None = None,
        purchase_method: AssetPurchaseMethod | str | None = None,
        priority: int = 999
    ) -> 'AssetAllocationParams':
        """
        Args:
            strategy (AssetAllocationStrategy | str): Allocation strategy for this allocation group:
               - NO_TRADE
               - BUY_HOLD
               - BUY_SELL
               - UNCONSTRAIN
            weight_box (tuple[float, float, float] | None): Target weight for the allocation group;
                a three-element tuple representing (lower, target, upper).
            primary_objective (dict[str, str | AssetRebalanceObjective | None] | PrimaryRebalanceObjective | None):
                Primary rebalance objectives;
                possible dict keys (str):
                    - below_range
                    - within_range
                    - above_range
                possible dict values (str | AssetRebalanceObjective):
                    - NOT_USED ('None' is concerted to 'NOT_USED')
                    - NO_ACTION
                    - TO_TARGET_WEIGHT
                    - TO_LOWER_WEIGHT
                    - TO_UPPER_WEIGHT
            purchase_method (AssetPurchaseMethod | str): Method to purchase new asset for this allocation group:
                - SCALE_UP_EXISTING
                - PURCHASE_PROFILE
                - NOT_USED ('None' is concerted to 'NOT_USED')
            priority (int): The priority in which the allocation group will be processed, smaller integer represents
            higher priority.

        Returns:
            AssetAllocationParams
        """
        if isinstance(strategy, str):
            strategy = AssetAllocationStrategy[strategy.upper()]
        if purchase_method is None:
            purchase_method = AssetPurchaseMethod.NOT_USED
        elif isinstance(purchase_method, str):
            purchase_method = AssetPurchaseMethod[purchase_method.upper()]
        if primary_objective is None:
            primary_objective = PrimaryRebalanceObjective(below_range=AssetRebalanceObjective.NOT_USED,
                                                          within_range=AssetRebalanceObjective.NOT_USED,
                                                          above_range=AssetRebalanceObjective.NOT_USED)
        elif isinstance(primary_objective, PrimaryRebalanceObjective):
            pass
        else: # dict
            for key in ("below_range", "within_range", "above_range"):
                if key not in primary_objective:
                    primary_objective[key] = AssetRebalanceObjective.NOT_USED
            primary_objective = PrimaryRebalanceObjective.create(**primary_objective)

        return AssetAllocationParams(
            strategy=strategy,
            weight_box=weight_box,
            primary_objective=primary_objective,
            purchase_method=purchase_method,
            priority=priority,
        )

    def __post_init__(self):
        if not isinstance(self.strategy, AssetAllocationStrategy):
            raise TypeError(f"Invalid type of 'strategy': {type(self.strategy)}, expected 'AssetAllocationStrategy'.")

        if not isinstance(self.purchase_method, AssetPurchaseMethod):
            raise TypeError(f"Invalid type of 'purchase_method': {type(self.purchase_method)}, expected 'AssetPurchaseMethod'.")

        if self.weight_box is None:
            pass
        elif isinstance(self.weight_box, tuple):
            if len(self.weight_box) != 3:
                raise ValueError(f"Invalid len of 'weight_box': {len(self.weight_box)}, expected 3.")
            if not all(isinstance(x, float) for x in self.weight_box):
                raise TypeError(f"Invalid element type of 'weight_box' ({self.weight_box}), expected 'float'.")
            if not self.weight_box[0] <= self.weight_box[1] <= self.weight_box[2]:
                raise ValueError(f"Invalid 'weight_box' values: {self.weight_box}.")
        else:
            raise TypeError(f"Invalid type of 'weight_box': {type(self.weight_box)}, expected 'tuple'.")

        if not isinstance(self.primary_objective, PrimaryRebalanceObjective):
            raise TypeError(f"Invalid type of 'primary_objective': {type(self.primary_objective)}, expected 'PrimaryRebalanceObjective'.")

class _AllocationGroup:
    """Asset allocation group.
    """

    __slots__ = ("name", "strategy", "primary_objective", "purchase_method", "purchase_method", "priority",
                 "exist_conn", "profile_conn", "staged_conn", "size_basis", "size_box",
                 "profile_size", "_trade_orders", "_free_proceeds")

    def __init__(
        self,
        *,
        name: str,
        exist_assets: Connector[Asset] | list[Asset],
        strategy: AssetAllocationStrategy,
        weight_box: tuple[float, float, float] | None,
        primary_objective: PrimaryRebalanceObjective,
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
        self.primary_objective: PrimaryRebalanceObjective = primary_objective
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

        self.size_box: tuple[float, float, float] = self._compute_size_box(total_size, weight_box)
        self.profile_size: float = self.profile_conn.sum_attr(self.size_basis) if self.profile_conn is not None else 0.0

        self.staged_conn: Connector[Asset] = Connector[Asset]()
        self._trade_orders: list[TradeOrder] = []
        self._free_proceeds: float = 0.0

        # initial validation
        if self.strategy in (AssetAllocationStrategy.BUY_HOLD, AssetAllocationStrategy.BUY_SELL):
            if self.purchase_method == AssetPurchaseMethod.NOT_USED:
                raise ValueError(f"'{self.name}': '{self.strategy.name}'; 'purchase_method' is 'NOT_USED'.")
            if self.purchase_method == AssetPurchaseMethod.SCALE_UP_EXISTING and len(self.exist_conn) == 0:
                raise ValueError(f"'{self.name}': 'SCALE_UP_EXISTING'; no existing asset.")
            if self.purchase_method == AssetPurchaseMethod.PURCHASE_PROFILE and len(self.profile_conn) == 0:
                raise ValueError(f"'{self.name}': 'PURCHASE_PROFILE';  no profile asset.")

    def _compute_size_box(self, total_size: float, weight_box: tuple[float, float, float] | None
                          ) -> tuple[float, float, float]:
        strategy = self.strategy
        if strategy == AssetAllocationStrategy.UNCONSTRAIN:
            return float("-inf"), 0.0, float("inf")
        if strategy == AssetAllocationStrategy.NO_TRADE:
            exist_size = self.exist_size
            return float("-inf"), exist_size, float("inf")
        if weight_box is None:
            raise ValueError(f"'{self.name}': strategy '{strategy}': 'weight_box' is None.")
        if strategy == AssetAllocationStrategy.BUY_HOLD:
            return total_size * weight_box[0], total_size * weight_box[1], float("inf")
        if strategy == AssetAllocationStrategy.BUY_SELL:
            return total_size * weight_box[0], total_size * weight_box[1], total_size * weight_box[2]
        raise ValueError("Should never get here.")

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
    def exist_market_value(self) -> float:
        return self.exist_conn.sum_attr("market_value")

    @property
    def staged_size(self) -> float:
        return self.staged_conn.sum_attr(self.size_basis)

    @property
    def primary_objective_size(self) -> float | None:
        strategy = self.strategy
        exist_size = self.exist_size
        lower_size, target_size, upper_size = self.size_box

        if strategy == AssetAllocationStrategy.UNCONSTRAIN:
            return None
        if strategy == AssetAllocationStrategy.NO_TRADE:
            return exist_size

        if strategy == AssetAllocationStrategy.BUY_HOLD:
            if exist_size >= lower_size:
                return exist_size
            prim_obj = self.primary_objective.below_range
            if prim_obj == AssetRebalanceObjective.TO_LOWER_WEIGHT:
                return lower_size
            elif prim_obj == AssetRebalanceObjective.TO_TARGET_WEIGHT:
                return target_size
            else:
                warnings.warn(f"strategy 'BUY_HOLD', below lower, primary objective: {prim_obj.name}, "
                              f"cannot determine size.")
                return exist_size

        if strategy == AssetAllocationStrategy.BUY_SELL:
            if exist_size < lower_size:
                prim_obj = self.primary_objective.below_range
                if prim_obj == AssetRebalanceObjective.TO_LOWER_WEIGHT:
                    return lower_size
                elif prim_obj == AssetRebalanceObjective.TO_TARGET_WEIGHT:
                    return target_size
                elif prim_obj == AssetRebalanceObjective.TO_UPPER_WEIGHT:
                    return upper_size
                else:
                    warnings.warn(f"strategy 'BUY_SELL', below range, primary objective: {prim_obj.name}, "
                                  f"cannot determine size.")
                    return exist_size
            elif exist_size > upper_size:
                prim_obj = self.primary_objective.above_range
                if prim_obj == AssetRebalanceObjective.TO_LOWER_WEIGHT:
                    return lower_size
                elif prim_obj == AssetRebalanceObjective.TO_TARGET_WEIGHT:
                    return target_size
                elif prim_obj == AssetRebalanceObjective.TO_UPPER_WEIGHT:
                    return upper_size
                else:
                    warnings.warn(f"strategy 'BUY_SELL', above range primary objective: {prim_obj.name}, "
                                  f"cannot determine size.")
                    return exist_size
            else:  # lower_size <= exist_size <= upper_size
                prim_obj = self.primary_objective.within_range
                if prim_obj == AssetRebalanceObjective.NO_ACTION:
                    return exist_size
                elif prim_obj == AssetRebalanceObjective.TO_LOWER_WEIGHT:
                    return lower_size
                elif prim_obj == AssetRebalanceObjective.TO_TARGET_WEIGHT:
                    return target_size
                elif prim_obj == AssetRebalanceObjective.TO_UPPER_WEIGHT:
                    return upper_size
                else:
                    warnings.warn(f"strategy 'BUY_SELL', within range, primary objective: {prim_obj.name}, "
                                  f"cannot determine size.")
                    return exist_size
        raise ValueError("Should never get here")

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
    def merge_trade_orders(cls, trade_orders: list[TradeOrder], *, on_exception_raise: bool) -> list[TradeOrder]:
        if len(trade_orders) == 1:
            return trade_orders
        tmp_dict = {x: 0.0 for x in BuySellAction}
        for order in trade_orders:
            tmp_dict[order.action] += order.arg
        merged: list[TradeOrder] = []
        for act, arg in tmp_dict.items():
            if act == BuySellAction.PASS:
                pass
            elif act == BuySellAction.EXCEPTION:
                if on_exception_raise:
                    raise ValueError("BuySellAction is EXCEPTION.")
                warnings.warn("BuySellAction is EXCEPTION.")
                pass
            elif abs(arg) > 0.0:
                merged.append(TradeOrder(action=act, arg=arg))
        return merged

    def execute_trades(self, trade_orders: TradeOrder | list[TradeOrder] | None = None, *, on_exception_raise: bool) -> None:
        if trade_orders is None:
            trade_orders = self._trade_orders
            consume_self_trade_orders = True
        else:
            consume_self_trade_orders = False
        trade_orders = trade_orders if isinstance(trade_orders, list) else [trade_orders]
        if len(trade_orders) == 0:
            return

        for order in self.merge_trade_orders(trade_orders, on_exception_raise=on_exception_raise):
            self.execute_trade(order, on_exception_raise=on_exception_raise)
        if consume_self_trade_orders:
            self._trade_orders = []  # reset after consumed

    def execute_trade(self, trade_order: TradeOrder, *, on_exception_raise: bool) -> None:
        """Execute a trade.
        """
        net_proceeds: float = 0.0  # proceeds received from disposal (sell), minus proceeds spent to purchase (buy)
        action, arg = trade_order.action, trade_order.arg

        if action == BuySellAction.PASS:
            pass
        elif action == BuySellAction.EXCEPTION:
            if on_exception_raise:
                raise ValueError("BuySellAction is EXCEPTION.")
            warnings.warn("BuySellAction is EXCEPTION.")
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
            for profile_asset in self.profile_conn:
                func = getattr(profile_asset, action.value)
                new_asset = func(scale=arg)
                self.staged_conn.append(new_asset) # append to asset list of the staged
                net_proceeds -= new_asset.market_value

        self._free_proceeds += net_proceeds

    def is_allocation_satisfied(self, rel_tol: float = 1e-4) -> tuple[bool, tuple[float, float]]:
        """Validate if the allocation for a group is satisfied.

        Args:
            rel_tol (float): Tolerance relative to target size for validation (default 0.0001).

        Returns:
            bool: True if the allocation is satisfied, False otherwise.
            tuple[float, float]: Reference sizes (lower, upper)
        """
        lower, target, upper = self.size_box

        if self.strategy in (AssetAllocationStrategy.UNCONSTRAIN, AssetAllocationStrategy.NO_TRADE):
            satisfied = True
        elif self.strategy == AssetAllocationStrategy.BUY_HOLD:
            satisfied = lower - rel_tol * target <= self.current_size
        elif self.strategy == AssetAllocationStrategy.BUY_SELL:
            satisfied = lower - rel_tol * target <= self.current_size <= upper + rel_tol * target
        else:
            raise ValueError(f"{self.name}: invalid asset allocation strategy {self.strategy}.")

        return satisfied, (lower, upper)