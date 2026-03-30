@echo off
cd /d "%~dp0"
cd ../..

FOR %%f IN (./examples/scripts/script_*.py) DO (
    echo Running %%f ...
    python "./examples/scripts/%%f"
)
