"""Regression tests for the `time` / `period` setters, the underlying
`ProjectionTimeSynchronizer`, and the `add_projection_time_synchronizer`
class decorator.

The setters are the pair most likely to regress: the `time` setter recently
carried an inverted boundary comparison, and both drain through the same
synchronizer. The tests pin the *inclusive* endpoint semantics explicitly.
"""

import gc

import pandas as pd
import pytest
from pandas._libs.tslibs.parsing import DateParseError

import vates._core._utils as _utils
from vates._core._utils import (
    ProjectionTimeSynchronizer,
    add_projection_time_synchronizer,
)



class TestTimeSetter:
    def test_endpoint_zero_accepted(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.time = 0
        assert m.time == 0
        assert m.period == m.START_DATE

    def test_endpoint_max_t_accepted(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.time = m.MAX_T
        # MAX_T is inclusive: the last period is exactly END_DATE.
        assert m.time == m.MAX_T
        assert m.period == m.END_DATE

    def test_interior_accepted(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.time = 5
        assert m.time == 5
        assert m.period == m.START_DATE + 5

    @pytest.mark.parametrize("bad", ["5", 5.0])
    def test_non_int_rejected(self, make_configured, tmp_path, bad):
        m = make_configured(tmp_path)
        with pytest.raises(TypeError):
            m.time = bad

    @pytest.mark.parametrize("bad", [-1, "placeholder_MAXT_PLUS_1"])
    def test_out_of_range_int_rejected(self, make_configured, tmp_path, bad):
        m = make_configured(tmp_path)
        with pytest.raises(ValueError):
            m.time = bad if isinstance(bad, int) else m.MAX_T + 1


class TestPeriodSetter:
    def test_endpoint_start_accepted(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.period = m.START_DATE
        assert m.period == m.START_DATE
        assert m.time == 0

    def test_endpoint_end_accepted(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.period = m.END_DATE
        assert m.period == m.END_DATE
        assert m.time == m.MAX_T

    def test_interior_period_accepted(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.period = m.START_DATE + 5
        assert m.time == 5

    def test_interior_str_coerced(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        # multi-year window: START 2026-12, END 2028-12 -> 2027-06 is interior
        m.period = "2027-06"
        assert m.period == pd.Period("2027-06", freq="M")
        assert m.time == 6

    def test_bare_non_period_rejected_typeerror(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        with pytest.raises(TypeError):
            m.period = 5

    def test_unparseable_string_raises_dateparseerror(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        with pytest.raises(DateParseError):
            m.period = "not-a-date"

    def test_before_start_rejected(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        with pytest.raises(ValueError):
            m.period = "2025-01"  # before START_DATE 2026-12

    def test_after_end_rejected(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        with pytest.raises(ValueError):
            m.period = "2029-01"  # after END_DATE 2028-12


class TestTimePeriodSymmetry:
    def test_setting_time_sets_period(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        for t in (0, 3, m.MAX_T):
            m.time = t
            assert m.period == m.START_DATE + t
            assert m.time == t

    def test_setting_period_sets_time(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        for t in (0, 3, m.MAX_T):
            m.period = m.START_DATE + t
            assert m.time == t
            assert m.period == m.START_DATE + t

    def test_boundaries_inclusive_not_half_open(self, make_configured, tmp_path):
        # Lock in the inclusive comparison (`<=`, not `<`). The `time` setter was
        # previously inverted; the `period` setter must allow its own boundaries too.
        m = make_configured(tmp_path)
        m.time = m.MAX_T  # must not raise
        assert m.time == m.MAX_T
        m.time = 0  # must not raise
        assert m.time == 0
        m.period = m.END_DATE  # must not raise
        assert m.period == m.END_DATE
        m.period = m.START_DATE  # must not raise
        assert m.period == m.START_DATE


class TestPreConfigGuardedState:
    def test_setting_time_before_config_raises(self, make_engine):
        m = make_engine()
        # Both setters read START_DATE/MAX_T/END_DATE from `_run_config`, which
        # does not exist until configure_run. A missing member surfaces as
        # AttributeError here (not a clean ValueError).
        with pytest.raises(AttributeError):
            m.time = 0

    def test_setting_period_before_config_raises(self, make_engine):
        m = make_engine()
        with pytest.raises(AttributeError):
            m.period = "2026-01"


class TestProjectionTimeSynchronizer:
    def test_defaults_none(self):
        s = ProjectionTimeSynchronizer()
        assert s.time is None
        assert s.period is None

    def test_set_time_only(self):
        s = ProjectionTimeSynchronizer()
        s.set(time=7)
        assert s.time == 7
        assert s.period is None

    def test_set_period_string_coerced(self):
        s = ProjectionTimeSynchronizer()
        s.set(period="2027-03")
        assert s.period == pd.Period("2027-03", freq="M")

    def test_elapse_increments_both(self):
        s = ProjectionTimeSynchronizer(time=0, period=pd.Period("2026-12", freq="M"))
        s.elapse(3)
        assert s.time == 3
        assert s.period == pd.Period("2027-03", freq="M")

    def test_observer_notified(self):
        calls = []

        class Observer:
            def _update_on_time_change(self):
                calls.append(self.time)

        s = ProjectionTimeSynchronizer()
        obs = Observer()
        s.attach_eligible_time_observer(obs)
        s.set(time=1)
        s.set(time=2)
        assert calls == [1, 2]

    def test_detach_time_observer_removes(self):
        s = ProjectionTimeSynchronizer()
        observer = self._Observer()
        s.attach_eligible_time_observer(observer)
        assert len(s._time_observers) == 1
        s.detach_time_observer(observer)
        assert len(s._time_observers) == 0
        s.set(time=1)
        assert observer.calls == 0  # no longer notified

    def test_detach_time_observer_absent_noop(self):
        s = ProjectionTimeSynchronizer()
        observer = self._Observer()  # hold a strong reference so it stays alive
        s.attach_eligible_time_observer(observer)
        s.detach_time_observer(self._Observer())  # different instance -> not found
        assert len(s._time_observers) == 1
        # detaching from an empty synchronizer is harmless
        s2 = ProjectionTimeSynchronizer()
        s2.detach_time_observer(self._Observer())
        assert len(s2._time_observers) == 0

    def test_duplicate_attach_ignored_silently(self):
        # Re-attaching the same object is silently skipped (hard dedup, no
        # warning); the original single registration remains and still receives
        # notifications.
        s = ProjectionTimeSynchronizer()
        observer = self._Observer()
        s.attach_eligible_time_observer(observer)
        s.attach_eligible_time_observer(observer)
        assert len(s._time_observers) == 1  # duplicate ignored
        s.set(time=1)
        assert observer.calls == 1

    class _Observer:
        def __init__(self):
            self.calls = 0

        def _update_on_time_change(self):
            self.calls += 1

    def test_dead_observer_dropped_automatically(self):
        # `WeakSet` drops dead observers eagerly, with no threshold heuristic.
        # Live observers must be strongly referenced from outside.
        s = ProjectionTimeSynchronizer()

        dead = self._Observer()
        s.attach_eligible_time_observer(dead)
        assert len(s._time_observers) == 1

        live = [self._Observer() for _ in range(4)]
        for observer in live:
            s.attach_eligible_time_observer(observer)
        assert len(s._time_observers) == 5

        del dead
        gc.collect()
        assert len(s._time_observers) == 4  # dead one gone, live four remain

    def test_dead_observer_not_notified(self):
        # An observer with no other reference is collected, so it is never
        # notified and no longer counted.
        s = ProjectionTimeSynchronizer()
        s.attach_eligible_time_observer(self._Observer())
        gc.collect()  # the observer has no other reference after the call returns
        s.set(time=1)
        assert len(s._time_observers) == 0


class TestAddProjectionTimeSynchronizer:
    """The class decorator injects `_time_synchronizer` and caches `time`/`period`.

    Mirrors `vates/alm/assets/asset_base.py` (the canonical user): a bare
    `@add_projection_time_synchronizer`, a keyword-only `model_engine` parameter,
    and `time`/`period` cached as plain instance attributes refreshed through the
    shared synchronizer.
    """

    def test_bare_decorator_wires_to_engine_synchronizer(self, make_configured, tmp_path):
        @add_projection_time_synchronizer
        class Asset:
            pass

        m = make_configured(tmp_path)
        asset = Asset(model_engine=m)
        # same synchronizer object, so a read reflects the engine's state
        assert asset._time_synchronizer is m.time_synchronizer
        assert asset.time is None
        assert asset.period is None

        m.time = 3
        assert asset.time == 3
        assert asset.period == m.START_DATE + 3

    def test_engine_period_setter_propagates_to_asset(self, make_configured, tmp_path):
        @add_projection_time_synchronizer
        class Asset:
            pass

        m = make_configured(tmp_path)
        asset = Asset(model_engine=m)
        m.period = m.START_DATE + 6
        assert asset.period == m.START_DATE + 6
        assert asset.time == 6

    def test_no_engine_raises(self, monkeypatch):
        monkeypatch.setattr(_utils, "FALLBACK_TIME_SYNCHRONIZER", None)

        @add_projection_time_synchronizer
        class Asset:
            pass

        with pytest.raises(ValueError, match="Failed to add projection time synchronizer"):
            Asset()
        with pytest.raises(ValueError, match="Failed to add projection time synchronizer"):
            Asset(model_engine=None)

    def test_model_engine_without_synchronizer_raises(self, monkeypatch):
        # `model_engine` must expose a `time_synchronizer` attribute; a plain
        # object does not, so the fallback/error branch is reached.
        monkeypatch.setattr(_utils, "FALLBACK_TIME_SYNCHRONIZER", None)

        @add_projection_time_synchronizer
        class Asset:
            pass

        with pytest.raises(ValueError, match="Failed to add projection time synchronizer"):
            Asset(model_engine=object())

    def test_fallback_synchronizer_used(self, monkeypatch):
        sync = ProjectionTimeSynchronizer(time=2, period=pd.Period("2026-12", freq="M"))
        monkeypatch.setattr(_utils, "FALLBACK_TIME_SYNCHRONIZER", sync)

        @add_projection_time_synchronizer
        class Asset:
            pass

        asset = Asset()
        assert asset.time == 2
        assert asset.period == pd.Period("2026-12", freq="M")

    def test_original_init_still_runs(self, make_configured, tmp_path):
        # The class's own __init__ is preserved and re-invoked with the same
        # kwargs (asset_base pattern: keyword-only model_engine parameter).
        @add_projection_time_synchronizer
        class Asset:
            def __init__(self, *, model_engine=None, label=None):
                self.label = label

        m = make_configured(tmp_path)
        asset = Asset(model_engine=m, label="eq")
        assert asset.label == "eq"
        m.time = 1
        assert asset.time == 1

    def test_parentheses_form_equivalent(self, make_configured, tmp_path):
        # `add_projection_time_synchronizer()` (no args) returns the decorator
        # factory; applying it yields the same behavior as the bare form.
        Asset = add_projection_time_synchronizer()(type("Asset", (), {}))

        m = make_configured(tmp_path)
        asset = Asset(model_engine=m)
        m.time = 4
        assert asset.time == 4
        assert asset.period == m.START_DATE + 4

    def test_observer_registered_and_notified(self, make_configured, tmp_path):
        # When the class defines `update_on_time_change`, the decorator registers
        # the instance and it is notified on every time change.
        recorded = []

        @add_projection_time_synchronizer
        class Asset:
            def _update_on_time_change(self):
                recorded.append(self.time)

        m = make_configured(tmp_path)
        asset = Asset(model_engine=m)
        assert len(m.time_synchronizer._time_observers) == 1
        m.time = 1
        m.time = 5
        assert recorded == [1, 5]
        assert len(m.time_synchronizer._time_observers) == 1  # still attached
        # the engine and the decorated object share the notify path
        assert asset.time == 5

    def test_time_period_cached_and_refreshed(self, make_configured, tmp_path):
        # A bare decorated class caches `time`/`period` as plain instance
        # attributes, refreshed on every time change.
        @add_projection_time_synchronizer
        class Asset:
            pass

        m = make_configured(tmp_path)
        asset = Asset(model_engine=m)
        assert len(m.time_synchronizer._time_observers) == 1
        m.time = 1
        assert asset.time == 1
        assert asset.period == m.START_DATE + 1
        assert "time" in asset.__dict__          # cached, not a class property
        assert "time" not in Asset.__dict__

    def test_mid_period_construction_seeds_cache(self, make_configured, tmp_path):
        # An object created after time has advanced reads the current time during
        # construction (spawned-asset case).
        @add_projection_time_synchronizer
        class Asset:
            pass

        m = make_configured(tmp_path)
        m.time = 7
        asset = Asset(model_engine=m)
        assert asset.time == 7
        assert asset.period == m.START_DATE + 7

    def test_class_time_attribute_shadowed_by_instance_cache(self, make_configured, tmp_path):
        # A class-level `time`/`period` is left untouched on the class, but the
        # instance caches the synchronizer values as instance attributes.
        @add_projection_time_synchronizer
        class Asset:
            time = "class-attr"
            period = "class-attr"

        m = make_configured(tmp_path)
        asset = Asset(model_engine=m)
        assert asset.time is None       # seeded from the (unset) synchronizer
        assert asset.period is None
        m.time = 3
        assert asset.time == 3
        assert asset.period == m.START_DATE + 3
        assert Asset.time == "class-attr"    # class attribute untouched
        assert Asset.period == "class-attr"

    def test_no_property_injected_on_class(self):
        # The decorator no longer injects `time`/`period` properties.
        @add_projection_time_synchronizer
        class Asset:
            pass

        assert "time" not in Asset.__dict__
        assert "period" not in Asset.__dict__