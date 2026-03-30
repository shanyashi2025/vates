@echo off
cd /d "%~dp0"
cd ../..

FOR %%f IN (./examples/cli/cli_*.bat) DO (
    echo Running %%f ...
    call "./examples/cli/%%f"
)
