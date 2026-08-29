"""Tests for projection result serialization: `.proj.csv` writing and `proj_result`
reading, plus the write toggles.

These tests need real bound projections, so they build a minimal model inline.
"""

import json

import pandas as pd
import pytest

from vates import ConstVariable, ProjModelEngine, TDimVariable, proj_result


class TestWriteToggles:
    def test_runlog_written_by_default(self, tmp_path):
        m = ProjModelEngine(model_name="m")
        m.configure_run(start_year=2026, end_year=2027, workspace_directory=str(tmp_path),
                        results_directory="res")
        m.bind_projection(lambda: None)
        m.run()
        assert (tmp_path / "res" / "m.runlog.json").is_file()

    def test_no_runlog_when_disabled(self, tmp_path):
        m = ProjModelEngine(model_name="m")
        m.configure_run(start_year=2026, end_year=2027, workspace_directory=str(tmp_path),
                        results_directory="res", enable_write_runlog=False)
        m.bind_projection(lambda: None)
        m.run()
        assert not (tmp_path / "res" / "m.runlog.json").exists()

    def test_runlog_json_content(self, tmp_path):
        m = ProjModelEngine(model_name="m")
        m.configure_run(start_year=2026, end_year=2027, workspace_directory=str(tmp_path),
                        results_directory="res")
        m.bind_projection(lambda: None)
        runlog = m.run()
        assert runlog["model_name"] == "m"
        assert runlog["execution"]["success"] is True
        assert "configuration" in runlog


def _bind_write_project(make_configured, tmp_path, model_name="m"):
    m = make_configured(tmp_path, model_name=model_name)

    def proj(model: ProjModelEngine):
        model.const[...] = 7.0
        model.tdim[model.time] = float(model.time)

    m.bind_projection(proj)
    m.const = ConstVariable("const", model_engine=m, owner="owner", group="group")
    m.tdim = TDimVariable("tdim", model_engine=m, owner="owner", group="group")
    m.run()
    return m


class TestProjResultWrite:
    def test_proj_csv_written_and_readable(self, make_configured, tmp_path):
        m = _bind_write_project(make_configured, tmp_path)
        csv_path = tmp_path / "results" / "base" / "m.proj.csv"
        assert csv_path.is_file()
        raw = pd.read_csv(csv_path)
        # header: group, owner, variable, constant, then one YYYYMM column per month
        assert list(raw.columns[:4]) == ["group", "owner", "variable", "constant"]
        assert "202612" in raw.columns  # START_DATE month
        assert "202812" in raw.columns  # END_DATE month

    def test_full_dataframe(self, make_configured, tmp_path):
        m = _bind_write_project(make_configured, tmp_path)
        df = proj_result(results_directory=tmp_path / "results" / "base", model_name="m")
        assert set(df.index.names) == {"group", "owner", "variable"}
        assert df.loc[("group", "owner", "const"), "constant"] == 7.0

    def test_single_cell_with_date(self, make_configured, tmp_path):
        m = _bind_write_project(make_configured, tmp_path)
        val = proj_result(results_directory=tmp_path / "results" / "base", model_name="m",
                          group="group", owner="owner", variable="tdim", date=202703)
        # t = 3 (2026-12 + 3 = 2027-03); model writes float(t)
        assert val == 3.0

    def test_single_cell_without_date_uses_const(self, make_configured, tmp_path):
        m = _bind_write_project(make_configured, tmp_path)
        val = proj_result(results_directory=tmp_path / "results" / "base", model_name="m",
                          group="group", owner="owner", variable="const")
        assert val == 7.0

    def test_missing_required_args_raises(self, make_configured, tmp_path):
        m = _bind_write_project(make_configured, tmp_path)
        with pytest.raises(IndexError, match="Missing"):
            proj_result(results_directory=tmp_path / "results" / "base", model_name="m",
                        group="group", owner="owner")  # no variable

    def test_missing_row_raises_lookup(self, make_configured, tmp_path):
        m = _bind_write_project(make_configured, tmp_path)
        with pytest.raises(LookupError):
            proj_result(results_directory=tmp_path / "results" / "base", model_name="m",
                        group="nope", owner="owner", variable="const")

    def test_no_proj_csv_when_disabled(self, make_configured, tmp_path):
        m = make_configured(tmp_path, model_name="m", scenario="base")
        # rebuild with enable_write_proj_result disabled
        m2 = ProjModelEngine(model_name="m")
        m2.configure_run(start_year=2026, end_year=2028, workspace_directory=str(tmp_path),
                         results_directory="results/base", enable_write_proj_result=False)
        m2.const = ConstVariable("const", model_engine=m2, owner="owner", group="group")

        def proj(model: ProjModelEngine):
            model.const[...] = 7.0

        m2.bind_projection(proj)
        m2.run()
        assert not (tmp_path / "results" / "base" / "m.proj.csv").exists()