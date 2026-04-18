def run_model(model_cls, stoch_cls = None):
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("args_json_file", type=str, help='args json file for model instance')
    args = parser.parse_args()

    with open(args.args_json_file, 'r', encoding='utf-8') as file:
        model_args = json.load(file)

    if stoch_cls:
        model_instance = stoch_cls(model_cls, **model_args)
    else:
        model_instance = model_cls(**model_args)

    model_instance.run()

def call_func(func):
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("args_json_file", type=str, help='args json file for model instance')
    args = parser.parse_args()

    with open(args.args_json_file, 'r', encoding='utf-8') as file:
        model_args = json.load(file)

    func(**model_args)
