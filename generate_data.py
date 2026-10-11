"""Generate synthetic historical sales data and save it as a CSV.

The data simulates a company that spends money on Instagram ads and
tracks monthly sales. The relationship is roughly linear plus random
seasonal noise, which is exactly what linear regression can learn.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n_months = 96

instagram_ad_spend = np.random.uniform(500, 4000, n_months)
seasonal = 1500 * np.sin(np.linspace(0, 8 * np.pi, n_months))

sales = (
    80 * instagram_ad_spend
    + 12000
    + seasonal
    + np.random.normal(0, 12000, n_months)
)
sales = np.maximum(sales, 0).round(2)
instagram_ad_spend = instagram_ad_spend.round(2)

df = pd.DataFrame(
    {
        "month": pd.date_range("2018-01-01", periods=n_months, freq="ME").strftime("%Y-%m"),
        "instagram_ad_spend": instagram_ad_spend,
        "sales": sales,
    }
)

df.to_csv("data/sales_data.csv", index=False)
print(f"Created data/sales_data.csv with {len(df)} rows (8 years of monthly data).")
print(df.head())