working in progress ...

# Model Input Reference


## Example Model 01: Asset Projection

1. assets_bond.csv

Contains data of bonds.

| No | Variable             | Data Type | Description                                                                                                                                                                                |
|----|----------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | asset_id             | str       | Asset identifier.                                                                                                                                                                          |
| 2  | allocation_group     | str       | Asset allocation group for rebalancing.                                                                                                                                                    |
| 3  | currency_id          | str       | Currency identifier.                                                                                                                                                                       |
| 4  | fund_id              | str       | Fund name the asset belonging to.                                                                                                                                                          |
| 5  | issue_date           | date      | Date of bond issue.                                                                                                                                                                        |
| 6  | maturity_date        | date      | Date of bond maturity.                                                                                                                                                                     |
| 7  | coupon_rate          | float     | Coupon rate p.a.                                                                                                                                                                           |
| 8  | coupon_freq          | int       | Frequency of coupon payment. The number 1, 2, 4, 12 or 0: 1=annual, 2=half-year, 4=quarter, 12=monthly, 0=zero-coupon.                                                                     |
| 9  | redemp_sched_id      | str       | Redemption schedule, 'none' if not used.                                                                                                                                                   |
| 10 | face_value           | float     | Face value.                                                                                                                                                                                |
| 11 | rf_curve_id          | str       | Specified risk-free curve.                                                                                                                                                                 |
| 12 | mv_price_dirty       | float     | Market value price (dirty).                                                                                                                                                                |
| 13 | market_spread        | float     | The spread added to the spot rate curve that makes the present value of cash flows equal to the market price.                                                                              |
| 14 | abv_price_dirty      | float     | Amortized book value price (accured interest included).                                                                                                                                    |
| 15 | units                | float     | Number of bonds.                                                                                                                                                                           |
| 16 | credit_band_id       | str       | Specified credit band name, 'none' if credit risk not used.                                                                                                                                |
| 17 | asset_classification | str       | Asset classification: 'FVTPL', 'FVOCI' or 'AC'.                                                                                                                                            |
| 18 | pre_calculation      | str       | Pre-calculation(s) before model projection: 'market_spread', 'market_price', 'coupon_rate', 'risk_neutralization'. Multiple calculations are joined by ';', and will be executed in order. |


