---
name: run-model
description: Run an actuarial model. Use when asked to run, execute, or perform a projection with an actuarial model in this workspace.
user-invocable: true
---

# Run an actuarial model

In this file, `<WORKSPACE>` means the repository root containing `manifest.json`; this skill is located at `<WORKSPACE>/.agents/skills/run-model/SKILL.md`.

## Steps

### Step 1: Identify the model to run

Read `name` and `description` for each model in `<WORKSPACE>/manifest.json` (`MANIFEST`) and match the user's request to a model.

- If the request is unclear, ask the user to clarify, **DO NOT** guess.
- If no model matches, tell the user; **DO NOT** guess. The manifest may be out of date.

For the selected model, resolve `script_path` as absolute or relative to `<WORKSPACE>`. Verify that it exists and is a `.py` file. If it does not, tell the user and pause until they resolve the path; do not select a different script.


### Step 2: Collect run configuration

Use the selected model's `run_config_json_schema` in `MANIFEST`. Resolve every property as follows: for a required property, ask the user to provide a value, present `default` to the user for reference when it has; for an optional property, ask whether they want to provide it or omit it.

Collect required values first, then discuss optional values. Two properties require special handling: if `properties` includes `workspace_directory`, set it to `<WORKSPACE>` without asking; if `properties` includes `results_directory` and the user provides a value, conventionally the value is expected to start with `results/` (i.e. a sub-directory inside `results/`), if not so ask the user whether `results/` should be prefixed. Validate supplied values against the declared types and constraints, including `minimum`, `maximum`, `enum`, and `items`; do not silently coerce invalid values. If a value is invalid, explain why and ask the user to correct it before proceeding.

Once all values and defaults are settled, show the complete effective configuration in a Markdown table and ask the user to confirm it before saving. Include every property that will be written, including defaults and `workspace_directory` when applicable. If the user changes anything, show the updated complete table and ask for confirmation again.

Do not save a configuration or start execution before the user confirms this table.


### Step 3: Save run configuration as a JSON file

Save the confirmed configuration as a JSON file in `<WORKSPACE>/runs/`. Create the directory if needed.

- File name: `_agent_<slug>_<yyyymmdd_hhmmss>.json`, using the effective `slug` value if the schema has that property; otherwise use `_agent_<yyyymmdd_hhmmss>.json`. Use a filesystem-safe slug (no path separators or invalid filename characters). If the target name already exists, choose a unique name rather than overwriting it.
- Content: JSON object containing the complete confirmed configuration, including any defaults used. Do not add properties that are not in the schema.

For example:

```json
{
    "slug": "asset_model",
    "start_year": 2024,
    "start_month": 12,
    "end_year": 2026,
    "scenario": "base",
    "input_directories": ["inputs/input_dir1", "inputs/input_dir2"],
    "results_directory": "results/base"
}
```

After saving, provide the config file path as a link.


### Step 4: Execute the model

Use this command shape, substituting the actual paths before showing or executing it. Select `.venv` or `venv` based on the workspace setup:

```powershell
& "<VENV_PATH>\Scripts\python.exe" "<SCRIPT_PATH>" "<CONFIG_PATH>"
```

- `<VENV_PATH>`: use the existing workspace virtual environment described in `<WORKSPACE>/AGENTS.md` (`.venv` or `venv`). If neither exists, follow `AGENTS.md` to create `.venv` and install dependencies as directed there.
- `<SCRIPT_PATH>`: the verified script path from step 1.
- `<CONFIG_PATH>`: the config file saved in step 3.

Quote paths in the PowerShell command so paths containing spaces work. Show the fully resolved command in a code block, then ask whether the user wants you to execute it. Do not execute it until the user explicitly agrees. If the user declines, stop here.

If the user agrees, run the command and inspect its exit code. Treat exit code `0` as necessary but not sufficient for success:

- If the process exits with a nonzero code, follow the failure path below.

- If `run_config_json_schema::properties` includes `results_directory`, check the model's run log in the effective results directory, using the model's documented naming convention (typically `<effective results_directory>/<effective slug>.runlog.json`). If `results_directory` was omitted, resolve its location using the model's documented/default behavior; do not assume the run succeeded just because the process exited with code `0`.
- Treat a missing run log, invalid JSON, or `execution::success` other than `true` as a failure and follow the failure path below.
- On success, report `execution::duration` (display `1s` if it is less than one second); link the path of `<effective slug>.runlog.json` and each path in `results`. If a path is missing or cannot be linked, say so explicitly.


**Failure path:** (a) provide the original error message, (b) perform preliminary troubleshooting only (about two minutes at most), and (c) offer best-effort suggestions. Do **NOT** change code, inputs, configuration, or the environment to fix a problem unless the user asks you to.


## Gotchas

- Per `<WORKSPACE>/AGENTS.md`, `<WORKSPACE>/results/` is **READ-ONLY**: never create, edit, or delete files there. The model itself may write its outputs when run.
