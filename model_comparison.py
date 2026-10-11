"""Stage 2: compare regression models on the multi-channel dataset.

Models compared:
  1. Multiple Linear Regression
  2. Ridge Regression
  3. Polynomial Regression (degree 2)
  4. Random Forest Regressor

Each is scored with 5-fold cross-validation (R2 and MAE) so the ranking is
robust, then the best model is exported with joblib for the Streamlit app.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

CSV_PATH = "data/sales_data_multi.csv"
MODEL_PATH = "output/sales_model.joblib"


def load_data():
    df = pd.read_csv(CSV_PATH)
    X = pd.get_dummies(df, columns=["season"], drop_first=True).drop(
        columns=["month", "sales"]
    )
    y = df["sales"]
    feature_names = list(X.columns)
    return X, y, feature_names


def build_models():
    return {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10),
        "Polynomial Regression (deg 2)": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lr", LinearRegression()),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42
        ),
    }


def main():
    X, y, feature_names = load_data()
    print("Features:", feature_names)
    print("Shape:", X.shape)

    results = []
    trained = {}
    for name, model in build_models().items():
        cv = cross_validate(
            model, X, y, cv=5,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
            return_train_score=False,
        )
        r2 = cv["test_r2"].mean()
        mae = -cv["test_mae"].mean()
        results.append({"Model": name, "CV R2 (mean±std)": f"{r2:.3f} ± {cv['test_r2'].std():.3f}", "CV MAE ($)": f"{mae:,.0f}"})
        model.fit(X, y)
        trained[name] = model

    table = pd.DataFrame(results).sort_values("CV MAE ($)")
    print("\n==============================")
    print("   CROSS-VALIDATION RESULTS")
    print("==============================")
    print(table.to_string(index=False))
    print("==============================")

    best_name = table.iloc[0]["Model"]
    best_model = trained[best_name]

    if best_name == "Random Forest":
        importances = pd.Series(
            best_model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
        print("\nRandom Forest feature importances:")
        print(importances.round(4).to_string())
        chart_labels = importances.index
        chart_values = importances.values
        chart_title = "Random Forest feature importances"
    elif best_name.startswith("Polynomial"):
        print(f"\nBest model is polynomial (linear coefficients not directly interpretable).")
        chart_labels = feature_names
        chart_values = np.zeros(len(feature_names))
        chart_title = "Polynomial model"
    else:
        coefs = pd.Series(best_model.coef_, index=feature_names).sort_values(
            key=np.abs, ascending=False
        )
        print("\nBest model coefficients (sales impact per unit spend):")
        print(coefs.round(4).to_string())
        print(f"\nIntercept: {best_model.intercept_:,.2f}")
        chart_labels = coefs.index
        chart_values = coefs.values
        chart_title = "Sales impact per $1 spent (Multiple Linear Regression)"

    plt.figure(figsize=(9, 5))
    plt.barh(chart_labels, chart_values, color="#1f77b4")
    plt.xlabel("Impact on monthly sales ($)")
    plt.title(chart_title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/coeffs_plot.png", dpi=150)
    print("\nSaved chart to output/coeffs_plot.png")

    joblib.dump({"model": best_model, "features": feature_names}, MODEL_PATH)
    print(f"\nSaved best model ({best_name}) to {MODEL_PATH}")
    print("Feature columns order:", feature_names)


if __name__ == "__main__":
    main()