@echo off
cd /d "%~dp0"
cd ../..
python .\examples\model\em12_stoch_ec_mvl.py .\examples\cli\model_args_12.json
echo Complete!