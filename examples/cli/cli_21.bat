@echo off
cd /d "%~dp0"
cd ../..
python .\examples\model\em21_smith_wilson.py .\examples\cli\model_args_21.json
echo Complete!