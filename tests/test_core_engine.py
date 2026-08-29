"""Tests for the `ProjModelEngine` public contract: the 5-step usage flow,
`configure_run` defaulting, `run()` guard states, and the `bind_projection`
annotation contract."""

import inspect

import pandas as pd
import pytest

from vates import ProjModelEngine


class TestConstruction:
    def test_model_name_and_description(self, make_engine):
        m = make_engine(model_name="my_model", description="my description")
        assert m.MODEL_NAME == "my_model"
        assert m._description == "my description"

    def test_engine_no_config_yet(self, make_engine):
        m = make_engine()
        assert not hasattr(m, "_run_config")
        assert m.runlog == {}


class TestConfigureRun:
    def test_double_configure_raises(self, make_engine):
        m = make_engine()
        m.configure_run(
            start_year=2026, end_year=2027, workspace_directory="C:\\work",
            results_directory="res")
        with pytest.raises(ValueError):
            m.configure_run(start_year=2026, end_year=2027)

    def test_default_end_year_equals_start_year(self, make_engine, tmp_path):
        m = make_engine()
        m.configure_run(start_year=2026, workspace_directory=str(tmp_path))
        assert m.END_YEAR == 2026
        assert m.START_YEAR == 2026

    def test_default_end_month_12(self, make_engine, tmp_path):
        m = make_engine()
        m.configure_run(start_year=2026, end_year=2027, workspace_directory=str(tmp_path))
        assert m.START_MONTH == 12
        assert m.END_MONTH == 12
        assert m.MAX_T == 12  # 2026-12 .. 2027-12

    def test_start_year_none_raises(self, make_engine):
        m = make_engine()
        with pytest.raises(ValueError):
            m.configure_run(start_year=None)

    def test_full_config_read_back(self, make_engine, tmp_path):
        m = make_engine()
        m.configure_run(
            start_year=2026, start_month=3, end_year=2028, end_month=6,
            scenario="esg", workspace_directory=str(tmp_path),
            results_directory="out")
        assert m.START_DATE == pd.Period("2026-03", freq="M")
        assert m.END_DATE == pd.Period("2028-06", freq="M")
        assert m.SCENARIO == "esg"
        assert m.MAX_T == 27


class TestRunGuardStates:
    def test_unbound_raises(self, make_engine):
        m = make_engine()
        with pytest.raises(ValueError, match="has not been bound"):
            m.run()

    def test_bound_but_unconfigured_raises(self, make_engine):
        m = make_engine()
        m.bind_projection(zero_arg)
        with pytest.raises(ValueError, match="has not been set"):
            m.run()

    def test_runs_t_0_to_max_t(self, make_configured, tmp_path):
        seen = []

        def proj(model: ProjModelEngine):
            seen.append((model.time, model.period))

        m = make_configured(tmp_path)
        m.bind_projection(proj)
        m.run()
        assert len(seen) == m.MAX_T + 1
        assert seen[0] == (0, m.START_DATE)
        assert seen[-1] == (m.MAX_T, m.END_DATE)
        # lockstep: t and (period - START_DATE) stay consistent all the way
        for t, p in seen:
            assert (p - m.START_DATE).n == t

    def test_runlog_has_execution_success(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.bind_projection(lambda: None)  # zero-arg binds as a plain function
        runlog = m.run()
        assert runlog["execution"]["success"] is True
        assert runlog["model_name"] == m.MODEL_NAME


class TestBindProjection:
    def test_return_self(self, make_engine):
        m = make_engine()
        assert m.bind_projection(zero_arg) is m

    def test_bind_then_run_zero_arg(self, make_configured, tmp_path):
        results = []

        def zero_arg_proj():
            results.append(1)

        m = make_configured(tmp_path)
        m.bind_projection(zero_arg_proj)
        assert len(results) == 0
        m.run()
        assert len(results) == m.MAX_T + 1

    def test_annotated_function_bound_as_method(self, make_configured, tmp_path):
        called_with_self = []

        def proj(model: ProjModelEngine):
            called_with_self.append(model)

        m = make_configured(tmp_path)
        m.bind_projection(proj)
        m.run()
        assert len(called_with_self) == m.MAX_T + 1
        assert all(m_ is m for m_ in called_with_self)

    def test_double_bind_raises(self, make_engine):
        m = make_engine()
        m.bind_projection(zero_arg)
        with pytest.raises(ValueError, match="already bound"):
            m.bind_projection(zero_arg)

    def test_non_callable_raises(self, make_engine):
        m = make_engine()
        with pytest.raises(ValueError, match="un-callable"):
            m.bind_projection(42)

    def test_class_raises(self, make_engine):
        m = make_engine()
        with pytest.raises(ValueError, match="Cannot bind a class"):
            m.bind_projection(ProjModelEngine)

    def test_annotated_with_unresolved_hint_raises(self, make_engine):
        m = make_engine()
        # a string/forward annotation referencing a name importable nowhere
        def proj(model: "TotallyUndefinedType"):
            pass

        with pytest.raises(ValueError, match="cannot be resolved"):
            m.bind_projection(proj)


class TestBindProjectionIssubclassSemantics:
    """The engine's annotation check uses `issubclass(type(self), annotated)`,
    so a base-typed engine can be bound with a function annotated with the base
    type, and a subclass-typed engine with the base type. A function annotated
    with a *subclass* cannot be bound to the base engine."""

    def test_subclass_annotated_cannot_bind_to_base(self, make_engine):
        class SubEngine(ProjModelEngine):
            pass

        def proj(model: SubEngine):
            pass

        m = make_engine()
        with pytest.raises(ValueError, match="not a subclass"):
            m.bind_projection(proj)

    def test_base_annotated_can_bind_to_subclass(self, tmp_path):
        class SubEngine(ProjModelEngine):
            pass

        def proj(model: ProjModelEngine):
            pass

        m = SubEngine(model_name="sub")
        m.configure_run(start_year=2026, end_year=2027, workspace_directory=str(tmp_path))
        m.bind_projection(proj)
        m.run()  # executes fine

    def test_unannotated_first_arg_raises(self, make_engine):
        def proj(model):
            pass

        m = make_engine()
        with pytest.raises(ValueError, match="must annotate"):
            m.bind_projection(proj)


class TestSetattrGuard:
    def test_init_assignments_pass_before_initialized(self):
        # `_initialized` is armed at the end of `__init__`, so internal names
        # assigned during construction do not trip the guard.
        m = ProjModelEngine(model_name="g", description="d")
        assert m._model_name == "g"

    def test_overwrite_class_member_raises(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        with pytest.raises(AttributeError, match="protected member"):
            m.MAX_T = 99  # class-level members are read-only once armed

    def test_add_underscore_member_raises(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        with pytest.raises(AttributeError, match="private member"):
            m._foo = 1

    def test_add_public_member_allowed_and_traced(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.user_value = 123
        assert m.user_value == 123
        assert any("Add member" in msg for msg in m._messages)

    def test_property_setters_still_reachable(self, make_configured, tmp_path):
        m = make_configured(tmp_path)
        m.time = 3  # a property with an `fset` passes through the guard
        assert m.time == 3


def zero_arg():
    pass