import numpy as np
import json
import os
import sys
from vates import LightModelSpace
from vates.utils import smith_wilson_extrap, convert_spot_to_fwrd


def _output_spot_curve(sw_curve, output_file_path):
    import csv
    with open(output_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        header = ['maturity'] + [x for x in sw_curve]
        writer.writerow(header)
        for i in range(120):
            row = [i+1] + [v['spot'][i] for k, v in sw_curve.items()]
            writer.writerow(row)

def _output_forward_curve(sw_curve, output_file_path):
    import csv
    with open(output_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        header = ['maturity'] + [x for x in sw_curve]
        writer.writerow(header)
        for i in range(120):
            row = [i+1] + [v['forward'][i] for k, v in sw_curve.items()]
            writer.writerow(row)

def _get_output_file_path(wsdir, output_folder, output_file_name):
    output_folder_path = os.path.join(wsdir, output_folder)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    return os.path.join(wsdir, output_folder, output_file_name)


def generate_smith_wilson_curve(workspace_directory: str, input_directories: list[str]):
    model = LightModelSpace(name='smith_wilson', description="Generate the Smith-Wilson extrapolated curve.")
    model.set_run_config(
        workspace_directory=workspace_directory,
        input_directories=input_directories,
    )

    market_curves = []
    sw_params = []
    sw_curves = {}
    sw_param_df = model.read_csv('sw_param.csv', index_col="curve_id")
    market_data_df = model.read_csv('market_data.csv')
    output_config_df = model.read_csv('_output_config.csv', index_col="item")

    for row in sw_param_df.index:
        market_data_id = sw_param_df.at[row, 'market_data_id']
        market_spot = market_data_df[market_data_id].values
        llp = sw_param_df.at[row, 'last_liquid_point']
        ufr = sw_param_df.at[row, 'ufr']
        convergence_point = sw_param_df.at[row, 'convergence_point']
        min_alpha = sw_param_df.at[row, 'min_alpha']

        sw_curves[row] = smith_wilson_extrap(
            rates=np.array(market_spot[:llp]),
            maturities=np.arange(1, llp + 1),
            ufr=ufr,
            convergence_point=convergence_point,
            min_alpha=min_alpha
        )

        market_curve = {
            'spot': market_spot,
            'forward': convert_spot_to_fwrd(np.insert(arr=market_spot, obj=0, values=0), "A")[1:]
        }
        market_curves.append(market_curve)

        sw_param = {
            'llp': llp,
            'ufr': ufr,
            'convergence_point': convergence_point,
            'alpha': sw_curves[row]['alpha'],
            'min_alpha': min_alpha
        }
        sw_params.append(sw_param)

    # output spot rates (.csv)
    output_folder, output_file_name = output_config_df.loc['spot_rates_csv', ['output_folder', 'output_file_name']]
    output_file_path = _get_output_file_path(model.workspace_directory, output_folder, output_file_name)
    _output_spot_curve(sw_curves, output_file_path)
    # output forward rates (.csv)
    output_folder, output_file_name = output_config_df.loc['forward_rates_csv', ['output_folder', 'output_file_name']]
    output_file_path = _get_output_file_path(model.workspace_directory, output_folder, output_file_name)
    _output_forward_curve(sw_curves, output_file_path)

def main():
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        kwargs = json.load(file)
    generate_smith_wilson_curve(**kwargs)

if __name__ == "__main__":
    main()
