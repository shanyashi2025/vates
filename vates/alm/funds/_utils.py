from dataclasses import dataclass, field

from vates.alm.assets import Asset, Cash
from vates.alm.liabs import Liab


@dataclass(frozen=True)
class ALContainer:
    """Asset liability container

    """
    assets: list[Asset] = field(default_factory=list)
    liabs: list[Liab] = field(default_factory=list)

    def raise_duplicate_error(self, scope: str = "all", /) -> None:
        scope = scope.lower()
        if scope in ("all", "asset", "assets"):
            assets_set = set(self.assets)
            if len(self.assets) != len(assets_set):
                dup_lst = [x for x in assets_set if self.assets.count(x) > 1]
                raise ValueError(f"{dup_lst} duplicate asset objects, including '{dup_lst[:min(5, len(dup_lst))]}'.")
        if scope in ("all", "liability", "liabilities", "liab", "liabs"):
            liabs_set = set(self.liabs)
            if len(self.liabs) != len(liabs_set):
                dup_lst = [x for x in liabs_set if self.liabs.count(x) > 1]
                raise ValueError(f"{dup_lst} duplicate liability objects, including '{dup_lst[:min(5, len(dup_lst))]}'.")

    def raise_profile_asset_error(self) -> None:
        for asset in self.assets:
            if asset.is_profile:
                raise ValueError(f"Unexpected profile asset '{asset}'")

    @property
    def first_seen_cash_asset(self) -> Cash | None:
        for asset in self.assets:
            if isinstance(asset, Cash):
                return asset
        return None
