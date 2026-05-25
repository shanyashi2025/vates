import vates as vt # vt.AutogradCell

def calculate_cash_flows(issue_date: int, maturity_date: int, coupon_rate: float, coupon_freq: int, 
                         face_value: float, valn_date: int) -> list[float] | None:
    if valn_date < issue_date or valn_date >= maturity_date:
        return None

    issue_y12m, maturity_y12m, valn_y12m = get_y12m(issue_date), get_y12m(maturity_date), get_y12m(valn_date)

    n_months = maturity_y12m - issue_y12m
    cash_flows = [0.0] * n_months
    if coupon_rate == 0:
        pass
    elif coupon_freq == 0:
        cash_flows[n_months - 1] = face_value * coupon_rate
    elif coupon_freq in (1, 2, 4, 12):
        coupon_interval = 12 // coupon_freq
        coupon_payment = face_value * coupon_rate / coupon_freq
        for i in range(n_months):
            month = i + 1
            if month % coupon_interval == 0:
                cash_flows[i] = coupon_payment
    else:
        raise ValueError(f"Invalid {coupon_freq=}, expected (1, 2, 4, 12)")
    cash_flows[n_months - 1] += face_value

    return cash_flows[valn_y12m - issue_y12m:]

def get_y12m(date_6dgt: int) -> int:
    y, m = divmod(date_6dgt, 100)
    return y * 12 + m

def calculate_dcf(cash_flows: list[float], spot_curve: list[float], spread: float) -> float:
    dcf = 0
    for i, cf in enumerate(cash_flows, 1):
        if abs(cf) > 1e-8:
            dcf += cf / (1 + spot_curve[i] + spread) ** (i / 12)
    return dcf

def calculate_total_market_value(valn_date: int, bonds: list[dict], spot_curve: list[float]):
    total_market_value = 0
    for bond in bonds:
        cash_flows = calculate_cash_flows(bond['issue_date'], bond['maturity_date'], bond['coupon_rate'],
                                          bond['coupon_freq'], bond['face_value'], valn_date)
        total_market_value += calculate_dcf(cash_flows, spot_curve, bond['market_spread'])
    return total_market_value

def spot_curve_interp(spot_curve_y: list) -> list:
    # simple monthly-interpolation method for illustration: assuming constant forward rate through the year
    n_points_y = len(spot_curve_y)
    fv_y = [(1 + spot_curve_y[i]) ** i for i in range(n_points_y)] # `fv[i] = (1 + spot[i])^i`
    fv_m = [1]
    for i in range(1, n_points_y):
        ret_m = (fv_y[i] / fv_y[i - 1]) ** (1 / 12)
        for _ in range(12):
            fv_m.append(fv_m[-1] * ret_m) # `fv[m] = fv[m-1] * (fv[i] / fv[i-1])^(1/12)`
    return [0 if i == 0 else fv_m[i] ** (12 / i) - 1 for i in range(len(fv_m))]

def main():
    print(f"This python script illustrates how to employ the `AutogradCell` class to implement the backpropagation "
          f"algorithm to perform sensitivity testing in a fast way.")

    # --- set up ---
    # (1) set up valuation date
    valn_date = 202412
    # (2) set up bond data
    bonds = [
        {'issue_date': 201803, 'maturity_date': 204803, 'coupon_rate': 0.0422, 'coupon_freq': 2,
          'face_value': 100, 'market_value': 143.47352, 'market_spread': 0.000098},
        {'issue_date': 202005, 'maturity_date': 207005, 'coupon_rate': 0.0373, 'coupon_freq': 2,
         'face_value': 100, 'market_value': 153.112638, 'market_spread': -0.0000514},
        {'issue_date': 202206, 'maturity_date': 202906, 'coupon_rate': 0.0275, 'coupon_freq': 1,
         'face_value': 100, 'market_value': 107.414911, 'market_spread': -0.0000102},
        {'issue_date': 201907, 'maturity_date': 203907, 'coupon_rate': 0.041, 'coupon_freq': 1,
         'face_value': 100, 'market_value': 128.900209, 'market_spread': 0.000994288},
        {'issue_date': 202111, 'maturity_date': 205111, 'coupon_rate': 0.0375, 'coupon_freq': 1,
         'face_value': 100, 'market_value': 134.660393, 'market_spread': 0.0014993},
        {'issue_date': 202309, 'maturity_date': 203309, 'coupon_rate': 0.0294, 'coupon_freq': 1,
         'face_value': 100, 'market_value': 110.745789, 'market_spread': 0.000496148},
        {'issue_date': 201907, 'maturity_date': 203907, 'coupon_rate': 0.041, 'coupon_freq': 1,
         'face_value': 100, 'market_value': 128.900209, 'market_spread': 0.000994288},
        {'issue_date': 202111, 'maturity_date': 205111, 'coupon_rate': 0.0375, 'coupon_freq': 1,
         'face_value': 100, 'market_value': 134.660393, 'market_spread': 0.0014993},
        {'issue_date': 202309, 'maturity_date': 203309, 'coupon_rate': 0.0294, 'coupon_freq': 1,
         'face_value': 100, 'market_value': 110.745789, 'market_spread': 0.000496148},
        {'issue_date': 202409, 'maturity_date': 202510, 'coupon_rate': 0.0148, 'coupon_freq': 0,
         'face_value': 100, 'market_value': 100.242000, 'market_spread': 0.004024148},
    ]
    # (3) set up spot curve
    spot_curve_input = [
        0, 0.01081423, 0.011374707, 0.011881485, 0.012960708, 0.014211998, 0.01514501, 0.016009559, 0.016590377,
        0.016765876, 0.016972176, 0.017344624, 0.017775376, 0.01824194, 0.018721133, 0.019188843, 0.019601912,
        0.019926558, 0.020151785, 0.020266497, 0.020265878, 0.020208573, 0.020118737, 0.020005317, 0.01987744,
        0.019744374, 0.019615512, 0.019500362, 0.019408569, 0.01934996, 0.019334329, 0.019361466, 0.019423387,
        0.019512716, 0.019621924, 0.019743264, 0.019868703, 0.019989877, 0.020098072, 0.020184216, 0.020238925,
        0.020270094, 0.020292779, 0.02030813, 0.020317336, 0.020321623, 0.020322254, 0.020320526, 0.020317772,
        0.020315361, 0.0203147
    ]
    # (4) set up sensitivity
    sens_input = {
        "10bps_up": 0.001,
        "10bps_dn": -0.001,
        "20bps_up": 0.002,
        "20bps_dn": -0.002,
        "30bps_up": 0.003,
        "30bps_dn": -0.003,
        "[5;10;15]_10bps_up": {5: 0.001, 10: 0.001, 15: 0.001},
        "[10;10;20]_20bps_dn": {10: -0.002, 15: -0.002, 20: -0.002}
    }
    sens_dict = {}
    len_spot_curve = len(spot_curve_input)
    for k, v in sens_input.items():
        if isinstance(v, dict):
            sens_dict[k] = [0] * len_spot_curve
            for kk, vv in v.items():
                sens_dict[k][int(kk)] = vv
        else:
            sens_dict[k] = [v] * len_spot_curve

    # --- model run approach ---
    print("\n--- model run approach ---")
    spot_curve = spot_curve_interp(spot_curve_input)
    mv_base = calculate_total_market_value(valn_date, bonds, spot_curve)
    mv_sens = {}
    print(f"-   (slow) run model for each scenario ...")
    for key, sens in sens_dict.items():
        print(f"  - run model for scenario '{key}' ...")
        spot_curve = spot_curve_interp([b + s for b, s in zip(spot_curve_input, sens)])
        mv_sens[key] = calculate_total_market_value(valn_date, bonds, spot_curve)
    print("\nSUMMARY: model run approach")
    print(f"{'scenario':^20}| {'mv':^10} | {'delta':^8} | {'delta%':^6}")
    print(f"{'base':<20}| {mv_base:>10.2f}")
    for key, mv in mv_sens.items():
        print(f"{key:<20}| {mv:>10.2f} | {mv - mv_base:>8.2f} | {(mv - mv_base) / mv_base:>6.2%}")

    # --- backpropagation approach ---
    print("\n--- backpropagation approach ---")
    spot_curve_cell = [vt.autograd.Cell(x) for x in spot_curve_input]
    spot_curve = spot_curve_interp(spot_curve_cell)
    back_mv_base = calculate_total_market_value(valn_date, bonds, spot_curve)
    print("-   backward pass: `.backward()`")
    back_mv_base.backward("mv")

    print("-   (fast) estimate using formula `delta mv = delta param * param sens` ...")
    back_mv_delta = {}
    for key, sens in sens_dict.items():
        back_mv_delta[key] = 0
        for rate_delta, cell in zip(sens, spot_curve_cell):
            if abs(rate_delta) > 1e-8 and isinstance(cell, vt.autograd.Cell):
                back_mv_delta[key] += rate_delta * cell.grad.get("mv", 0)

    print("\nSUMMARY: backpropagation approach")
    print(f"{'scenario':^20}| {'mv':^10} | {'delta':^8} | {'delta%':^6}")
    mv_base = back_mv_base.value
    print(f"{'base':<20}| {mv_base:>10.2f}")
    for key, mv_delta in back_mv_delta.items():
        print(f"{key:<20}| {mv_base + mv_delta:>10.2f} | {mv_delta:>8.2f} | {mv_delta / mv_base:>6.2%}")

    print(f"\n- trained parameter sensitivity is output to file 'tut_AutogradCell_param_sens.csv', "
          f"it can be referenced for more sensitivity scenarios")
    # with open('tut_autograd_param_sens.csv', 'w', newline='') as f:  # output
    #     f.writelines('maturity,value,sensitivity')
    #     for i, cell in enumerate(spot_curve_cell):
    #         f.writelines(f"\n{','.join([str(i), str(cell.value), str(cell.grad.get("mv", 0))])}")

if __name__ == '__main__':
    main()