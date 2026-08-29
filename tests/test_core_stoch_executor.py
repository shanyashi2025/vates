"""Tests for the `StochExecutor` contract.

The end-to-end tests spawn real worker processes (Windows: `spawn`), so the
projection function must live at module level (it is pickled to the workers).
They run with `max_workers=1` to keep the process pool minimal.
"""

import json
import warnings
from multiprocessing import cpu_count

import pandas as pd
import pytest

from vates import ProjModelEngine, StochExecutor, TDimVariable

MAX_WORKERS_DEFAULT = 1


def _stoch_proj(model: ProjModelEngine):
    """Projection that writes a time-dimensioned variable on the worker engine.

    The variable is created lazily at `t == 0`, on the worker's own
    `ProjModelEngine` instance (`StochExecutor` has no `include_proj_variable`),
    mirroring the `em11_monte_carlo` example.
    """
    if model.time == 0:
        model.balance = TDimVariable("balance", model_engine=model, owner="owner", group="group")
    model.balance[model.time] = float(model.time)  # deterministic per sim


def _make_executor(*, model_name="stoch", tmp_path, input_directories=None,
                   simulations="1-2", max_workers=MAX_WORKERS_DEFAULT):
    executor = StochExecutor(model_name=model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        executor.configure_run(
            start_year=2026,
            end_year=2027,
            simulations=simulations,
            workspace_directory=str(tmp_path),
            input_directories=input_directories,
            results_directory="results",
            max_workers=max_workers,
        )
    return executor


class TestConfigureRun:
    def test_simulations_parsed(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path, simulations="1-10,13")
        assert se.SIMULATIONS == list(range(1, 11)) + [13]

    def test_max_workers_defaults_to_one(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path, max_workers=None)
        assert se._run_config.max_workers == 1

    def test_max_workers_clamped_to_cpu_count(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path, max_workers=9999)
        assert se._run_config.max_workers == cpu_count()

    def test_non_int_max_workers_reset_to_one(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path, max_workers="lots")
        assert se._run_config.max_workers == 1

    def test_non_positive_max_workers_reset_to_one(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path, max_workers=0)
        assert se._run_config.max_workers == 1

    def test_double_configure_raises(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="already set"):
            se.configure_run(simulations="1-2", start_year=2026)


class TestBindProjection:
    def test_zero_arg_rejected(self, tmp_path):
        def proj_no_arg():
            pass

        se = _make_executor(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="has no argument"):
            se.bind_projection(proj_no_arg)

    def test_unannotated_first_arg_rejected(self, tmp_path):
        def proj_unann(model):
            pass

        se = _make_executor(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="must annotate"):
            se.bind_projection(proj_unann)

    def test_non_proj_engine_annotation_rejected(self, tmp_path):
        def proj_bad(model: int):
            pass

        se = _make_executor(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="must be a 'ProjModelEngine'"):
            se.bind_projection(proj_bad)

    def test_valid_projection_bound(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path)
        assert se.bind_projection(_stoch_proj) is se
        assert se._proj_cls is ProjModelEngine

    def test_double_bind_raises(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path)
        se.bind_projection(_stoch_proj)
        with pytest.raises(ValueError, match="already bound"):
            se.bind_projection(_stoch_proj)


class TestRunGuardStates:
    def test_unbound_raises(self, tmp_path):
        se = _make_executor(tmp_path=tmp_path)
        with pytest.raises(ValueError, match="has not been bound"):
            se.run()

    def test_unconfigured_raises(self, tmp_path):
        se = StochExecutor(model_name="stoch")
        se.bind_projection(_stoch_proj)
        with pytest.raises(ValueError, match="has not been set"):
            se.run()


class TestBatches:
    def test_split_into_batches(self):
        se = StochExecutor(model_name="stoch")
        batches = se._create_batches(list(range(1, 11)), 3)
        assert batches == [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10)]

    def test_single_batch_when_one_worker(self):
        se = StochExecutor(model_name="stoch")
        batches = se._create_batches([1, 2, 3], 1)
        assert batches == [(1, 2, 3)]


@pytest.mark.stoch
class TestEndToEndRun:
    def _write_stoch_input(self, tmp_path, *, statistic=None):
        inputs = tmp_path / "inputs"
        inputs.mkdir(exist_ok=True)
        (inputs / "__stoch_variables__.json").write_text(
            json.dumps({"group": {"include": "__ALL__"}}), encoding="utf-8"
        )
        setting = {}
        if statistic is not None:
            setting["statistic"] = statistic
        (inputs / "__stoch_setting__.json").write_text(
            json.dumps(setting), encoding="utf-8"
        )
        return inputs

    def test_run_produces_stoch_files(self, tmp_path):
        self._write_stoch_input(tmp_path, statistic=None)
        se = _make_executor(tmp_path=tmp_path, input_directories=["inputs"])
        se.bind_projection(_stoch_proj)
        runlog = se.run()
        assert runlog["execution"]["success"] is True

        files = {p.name for p in (tmp_path / "results").glob("stoch*")}
        assert "stoch.proj.csv" in files       # first simulation also writes .proj.csv
        assert "stoch.1.stoch.csv" in files    # per-batch .stoch.csv
        assert "stoch.runlog.json" in files
        assert not any("stat" in name for name in files)  # no statistic requested

    def test_stat_file_only_when_statistic_configured(self, tmp_path):
        self._write_stoch_input(tmp_path, statistic={"mean": "mean", "std": "std"})
        se = _make_executor(tmp_path=tmp_path, input_directories=["inputs"])
        se.bind_projection(_stoch_proj)
        runlog = se.run()
        assert runlog["execution"]["success"] is True

        results = tmp_path / "results"
        assert (results / "stoch.stoch.stat.csv").is_file()
        stat = pd.read_csv(results / "stoch.stoch.stat.csv")
        assert {"group", "owner", "variable"}.issubset(stat.columns)
        assert any("mean" in col or "std" in col for col in stat.columns)

    def test_no_stat_file_without_setting(self, tmp_path):
        # No `__stoch_setting__.json` at all -> statistic is None -> no stat file.
        self._write_stoch_input(tmp_path, statistic=None)
        (tmp_path / "inputs" / "__stoch_setting__.json").unlink()
        se = _make_executor(tmp_path=tmp_path, input_directories=["inputs"])
        se.bind_projection(_stoch_proj)
        se.run()
        assert not any(p.name.endswith("stoch.stat.csv") for p in (tmp_path / "results").iterdir())