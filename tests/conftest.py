"""Shared fixtures for the `vates._core` test suite."""

import warnings

import pytest

from vates import ConstVariable, ProjModelEngine, TDimVariable

# A multi-year window is required so that `max_t > 0`: several behaviors
# (e.g. the `time`/`period` "interior" values, and `TDimVariable` indexing of a
# non-zero `t`) cannot be exercised in a degenerate single-month run where
# `max_t == 0`.
DEFAULT_START_YEAR = 2026
DEFAULT_END_YEAR = 2028


@pytest.fixture(autouse=True)
def _silence_default_fill_warnings():
    """`configure_run`/`run` emit informational `UserWarning`s when filling
    defaults or resetting `time`. They are not errors; keep the run log clean."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


@pytest.fixture
def engine_cls():
    """Return the concrete `ProjModelEngine` class (default)."""
    return ProjModelEngine


@pytest.fixture
def make_engine(engine_cls):
    """Factory returning a fresh, un-configured engine with a unique name."""

    def _make(model_name="m", description="desc"):
        return engine_cls(model_name=model_name, description=description)

    return _make


@pytest.fixture
def make_configured(make_engine):
    """Factory returning an engine configured to write results into `tmp_path`.

    `configure_run` is called before any variable is constructed, because
    `TDimVariable`/`ConstVariable` read `model_engine._run_config` at construction
    time. The result directory is a subdirectory of the test's `tmp_path`, so no
    writes leak into the repo.
    """

    def _make(tmp_path, model_name="m", *, start_year=DEFAULT_START_YEAR,
              end_year=DEFAULT_END_YEAR, scenario="base"):
        engine = make_engine(model_name)
        results_dir = tmp_path / "results" / (scenario or "")
        with warnings.catch_warnings():
            # configure_run emits a UserWarning when it fills defaults
            warnings.simplefilter("ignore", UserWarning)
            engine.configure_run(
                start_year=start_year,
                end_year=end_year,
                scenario=scenario,
                workspace_directory=str(tmp_path),
                results_directory=str(results_dir),
            )
        return engine

    return _make


@pytest.fixture
def engine(make_configured):
    """A default configured engine writing into `tmp_path`."""
    return make_configured


@pytest.fixture
def add_variables():
    """Helper that attaches a scalar `ConstVariable` and a scalar `TDimVariable`
    to an engine (must be called after `configure_run`)."""

    def _add(engine, *, owner="owner", group="group",
             const_name="const_var", tdim_name="tdim_var"):
        cvar = ConstVariable(const_name, model_engine=engine, owner=owner, group=group)
        tvar = TDimVariable(tdim_name, model_engine=engine, owner=owner, group=group)
        return cvar, tvar

    return _add