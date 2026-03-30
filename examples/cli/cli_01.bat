@echo off
cd /d "%~dp0"
cd ../..
python .\examples\model\em01_asset_proj.py .\examples\cli\model_args_01.json
echo Complete!