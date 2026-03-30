@echo off
cd /d "%~dp0"
cd ../..
python .\examples\model\em11_stoch_monte_carlo.py .\examples\cli\model_args_11.json
echo Complete!