import json
import sys

def run_with_json_config(func: callable):
    """
    used for command `python {my_script.py} {config.json}`, JSON config file will be parsed by `sys.argv[1]`
    """
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    func(**kwargs)
