@echo off
cd /d "%~dp0"
cd ../..
python .\examples\model\em22_efficient_frontier.py .\examples\cli\model_args_22.json
echo Complete!