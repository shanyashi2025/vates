@echo off
cd /d "%~dp0"
cd ../..
python .\examples\model\em03_cross_proj.py .\examples\cli\model_args_03.json
echo Complete!