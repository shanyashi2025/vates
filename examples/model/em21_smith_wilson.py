import numpy as np
import os
from vates import ProjModelEngine
from vates.utils import smith_wilson_extrap, convert_spot_to_fwrd

def _plot_curve(sw_curves, market_curves, sw_params, output_file_path):
    import matplotlib.pyplot as plt

    n = len(sw_curves)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(24, 8 * n), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (curve_id, curve_data) in enumerate(sw_curves.items()):
        for j, curve_type in enumerate(['spot', 'forward']):
            ax = axes[i * 2 + j]

            title = f"{curve_id}: {curve_type} rates"
            extrap_rates = curve_data[curve_type]
            market_rates = market_curves[i][curve_type]
            llp = sw_params[i]['llp']
            ufr = sw_params[i]['ufr']
            t2 = sw_params[i]['convergence_point']
            alpha = sw_params[i]['alpha']
            min_alpha = sw_params[i]['min_alpha']

            # Market data
            ax.scatter(range(1, len(market_rates) + 1), market_rates * 100, color='r', s=50, zorder=5,
                       label='Market Data')

            # Extrapolated curve
            ax.plot(range(1, 101), extrap_rates[0:100] * 100, 'b', linewidth=2, label='Smith-Wilson Curve')

            # Reference lines
            ax.axhline(y=ufr * 100, color='g', linestyle='--', label=f'UFR ({ufr * 100:.2f}%)')
            ax.axvline(x=t2, color='g', linestyle='--', alpha=0.7, label=f'Convergence Point ({t2}Y)')
            ax.axvline(x=llp, color='r', linestyle='--', alpha=0.7, label=f'Last Liquid Point ({llp}Y)')

            # Alpha info (legend-only marker)
            ax.scatter([], [], color='none', label=f"α = {alpha:.6f} (min α = {min_alpha:.2f})")

            # Labels and title
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.set_xlabel('Maturity (Years)')
            ax.set_ylabel('Interest Rate (%)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_file_path, dpi=300)
    plt.close(fig)


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


def main(workspace_directory: str, input_directories: list[str]):
    # Generate the Smith-Wilson extrapolated curve.
    model_space = ProjModelEngine(
        model_name = 'smith_wilson',
        start_year = 2000,  # not uese
        start_month = 12,   # not used
        end_year = 2000,    # not used
        workspace_directory = workspace_directory,
        input_directories = input_directories,
    )

    market_curves = []
    sw_params = []
    sw_curves = {}
    sw_param_df = model_space.read_csv('sw_param', index_col="curve_id")
    market_data_df = model_space.read_csv('market_data')
    output_config_df = model_space.read_csv('_output_config', index_col="item")

    for row in sw_param_df.index:
        market_data_id = sw_param_df.loc[row, 'market_data_id']
        market_spot = market_data_df[market_data_id].values
        llp = sw_param_df.loc[row, 'last_liquid_point']
        ufr = sw_param_df.loc[row, 'ufr']
        convergence_point = sw_param_df.loc[row, 'convergence_point']
        min_alpha = sw_param_df.loc[row, 'min_alpha']

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

    # output plot (.png)
    output_folder, output_file_name = output_config_df.loc['plot_png', ['output_folder', 'output_file_name']]
    output_file_path = _get_output_file_path(model_space.workspace_directory, output_folder, output_file_name)
    _plot_curve(sw_curves, market_curves, sw_params, output_file_path)
    # output spot rates (.csv)
    output_folder, output_file_name = output_config_df.loc['spot_rates_csv', ['output_folder', 'output_file_name']]
    output_file_path = _get_output_file_path(model_space.workspace_directory, output_folder, output_file_name)
    _output_spot_curve(sw_curves, output_file_path)
    # output forward rates (.csv)
    output_folder, output_file_name = output_config_df.loc['forward_rates_csv', ['output_folder', 'output_file_name']]
    output_file_path = _get_output_file_path(model_space.workspace_directory, output_folder, output_file_name)
    _output_forward_curve(sw_curves, output_file_path)

if __name__ == "__main__":
    from vates import cli_main
    cli_main(main)
