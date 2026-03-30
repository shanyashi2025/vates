@echo off
cd /d "%~dp0"
cd ../..
python .\examples\model\em02_fund_proj.py .\examples\cli\model_args_02.json
echo Complete!