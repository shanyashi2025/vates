import json
import pandas as pd
from pathlib import Path
try:
    import matplotlib
    matplotlib.use("Agg")
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

from vates.finmath import extrapolate_interest_rates

def main():
    base_dir = Path(__file__).resolve().parent
    df = pd.read_csv(base_dir / "interest_rate_input.csv")
    rates_in = df["spot_rate"].values
    has_t0 = (int(df["T"].min()) == 0)

    with open(base_dir / "extrapolation_params.json", 'r', encoding='utf-8') as file:
        params_dict = json.load(file)

    for key, params in params_dict.items():
        method = params["method"]
        if method == "eiopa_alternative":
            params["llfr_weight"] = {(item["x"], item["y"]): item["w"] for item in params["llfr_weight"]}
        rates_out = extrapolate_interest_rates(rates_in, has_t0=has_t0, **params)
        df = pd.DataFrame(
            data={
                "discount": rates_out.discount,
                "spot":rates_out.spotac,
                "forward": rates_out.forwardac,
            },
            index=range(rates_out.max_maturity + 1)
        )
        df.to_csv(base_dir / f"{key}.csv", index=True)

        if has_matplotlib:
            import matplotlib.pyplot as plt
            ufr = params['ufr']

            # Extrapolated curve
            plt.plot(range(1, 101), rates_out.spotac[0:100] * 100, 'r', linewidth=2, label='Spot rate')
            plt.plot(range(1, 101), rates_out.forwardac[0:100] * 100, 'b', linewidth=2, label='Forward rate')

            # Reference lines
            plt.axhline(y=ufr * 100, color='g', linestyle='--', label=f'UFR ({ufr * 100:.2f}%)')

            # Labels and title
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.xlabel("Maturity (Years)", fontsize=7)
            plt.ylabel("Interest Rate (%)", fontsize=7)
            plt.title(f"Interest Rate Extrapolation - {params["method"]} : {key}", fontsize=8)

            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            plt.savefig(base_dir / f"{key}.png", dpi=300)
            plt.close()


if __name__ == "__main__":
    main()
