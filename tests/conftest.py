"""Shared fixtures for the `vates._core` test suite."""

import pytest

from vates import (
    ConstArrayVariable,
    ProjModelEngine,
    ScalarVariable,
    TDimArrayVariable,
    TDimVariable,
)

# A multi-year window is required so that `max_t > 0`: several behaviors
# (e.g. the `time`/`period` "interior" values, and `TDimVariable` indexing of a
# non-zero `t`) cannot be exercised in a degenerate single-month run where
# `max_t == 0`.
DEFAULT_START_YEAR = 2026
DEFAULT_END_YEAR = 2028


@pytest.fixture
def engine_cls():
    """Return the concrete `ProjModelEngine` class (default)."""
    return ProjModelEngine


@pytest.fixture
def make_engine(engine_cls):
    """Factory returning a fresh, un-configured engine with a unique name."""

    def _make(slug="m", description="desc"):
        return engine_cls(slug=slug, description=description)

    return _make


@pytest.fixture
def make_configured(make_engine):
    """Factory returning an engine configured to write results into `tmp_path`.

    `configure_run` is called before any variable is constructed, because
    Variables are created after configuration. The result directory is a
    subdirectory of the test's `tmp_path`, so no writes leak into the repo.
    """

    def _make(tmp_path, slug="m", *, start_year=DEFAULT_START_YEAR,
              end_year=DEFAULT_END_YEAR, scenario="base"):
        engine = make_engine(slug)
        results_dir = tmp_path / "results" / (scenario or "")
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
def engine(make_configured, tmp_path):
    """A default configured engine writing into `tmp_path`."""
    return make_configured(tmp_path)


@pytest.fixture
def add_variables():
    """Helper that attaches a scalar `ScalarVariable` and a scalar `TDimVariable`
    to an engine (must be called after `configure_run`)."""

    def _add(*, owner="owner", group="group",
             const_name="const_var", tdim_name="tdim_var"):
        cvar = ScalarVariable(const_name, owner=owner, group=group)
        tvar = TDimVariable(
            tdim_name,
            owner=owner,
            group=group,
            max_t=(DEFAULT_END_YEAR - DEFAULT_START_YEAR) * 12,
        )
        return cvar, tvar

    return _add


@pytest.fixture(autouse=True)
def clean_variable_cached():
    ScalarVariable._ng_cache = set()
    ConstArrayVariable._ng_cache = set()
    ConstArrayVariable._dims_cache = set()
    TDimVariable._ng_cache = set()
    TDimArrayVariable._ng_cache = set()
    TDimArrayVariable._dims_cache = set()
    yield