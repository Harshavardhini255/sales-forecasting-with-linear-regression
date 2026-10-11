"""Generate synthetic monthly marketing data with multiple channels.

Extends the single-feature dataset with TV ads, email spend and seasonal
effects, so we can run multiple linear regression and compare models.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n_months = 96

month_idx = range(n_months)
instagram_ad_spend = np.random.uniform(500, 4000, n_months).round(2)
tv_ad_spend = np.random.uniform(2000, 20000, n_months).round(2)
email_spend = np.random.uniform(200, 3000, n_months).round(2)

season_factor = {
    1: 0.92, 2: 1.00, 3: 1.10, 4: 1.18, 5: 1.05, 6: 0.95,
    7: 0.90, 8: 1.15, 9: 1.08, 10: 1.12, 11: 1.25, 12: 1.22,
}

months = pd.date_range("2018-01-01", periods=n_months, freq="ME").month
season_mult = np.array([season_factor[m] for m in months])

sales = (
    82 * instagram_ad_spend
    + 3.5 * tv_ad_spend
    + 1.2 * email_spend
    + 15000
) * season_mult + np.random.normal(0, 15000, n_months)
sales = np.maximum(sales, 0).round(2)

season_map = {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2",
              7: "Q3", 8: "Q3", 9: "Q3", 10: "Q4", 11: "Q4", 12: "Q4"}

df = pd.DataFrame(
    {
        "month": pd.date_range("2018-01-01", periods=n_months, freq="ME").strftime("%Y-%m"),
        "instagram_ad_spend": instagram_ad_spend,
        "tv_ad_spend": tv_ad_spend,
        "email_spend": email_spend,
        "season": [season_map[m] for m in months],
        "sales": sales,
    }
)

df.to_csv("data/sales_data_multi.csv", index=False)
print(f"Created data/sales_data_multi.csv with {len(df)} rows.")
print(df.head())