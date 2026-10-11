"""Predict sales from Instagram ad spend using linear regression.

Answers the question: "If we spend more on Instagram ads, how much will
sales increase?"
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CSV_PATH = "data/sales_data.csv"


def load_data():
    df = pd.read_csv(CSV_PATH)
    return df


def train_model(df):
    X = df[["instagram_ad_spend"]]
    y = df["sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2


def plot_results(df, model):
    plt.figure(figsize=(9, 5))
    plt.scatter(df["instagram_ad_spend"], df["sales"], alpha=0.6, label="Historical data")
    xs = pd.DataFrame({"instagram_ad_spend": [df["instagram_ad_spend"].min(), df["instagram_ad_spend"].max()]})
    ys = model.predict(xs)
    plt.plot(xs["instagram_ad_spend"], ys, color="red", linewidth=2, label="Regression line")
    plt.xlabel("Instagram ad spend ($)")
    plt.ylabel("Sales ($)")
    plt.title("Sales vs Instagram Ad Spend (Linear Regression)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/regression_plot.png", dpi=150)
    print("Saved plot to output/regression_plot.png")


def show_equation(model):
    intercept = model.intercept_
    slope = model.coef_[0]
    print("\n==============================")
    print("       MODEL EQUATION")
    print("==============================")
    print(f"sales = {intercept:,.2f} + {slope:,.2f} * instagram_ad_spend")
    print(f"\nInterpretation: for every extra $1 spent on Instagram ads,")
    print(f"sales increase by about ${slope:,.2f} on average.")
    print("==============================")


def predict(model, spend):
    row = pd.DataFrame({"instagram_ad_spend": [spend]})
    return model.predict(row)[0]


def predict_additional(model):
    print("\n-- Example forecasts --")
    for spend in [1500, 2500, 3500, 5000]:
        prediction = predict(model, spend)
        print(f"If you spend ${spend:,}/month on Instagram ads, predicted sales: ${prediction:,.2f}/month")


def main():
    df = load_data()
    model, X_train, X_test, y_train, y_test = train_model(df)
    mae, rmse, r2 = evaluate(model, X_test, y_test)

    print("Loaded", len(df), "months of data from", CSV_PATH)
    print("\n==============================")
    print("     MODEL PERFORMANCE")
    print("==============================")
    print(f"R2   (how well the model explains sales): {r2:.3f}  (1.0 = perfect)")
    print(f"MAE          (avg error in $): {mae:,.2f}")
    print(f"RMSE         (root mean sq err, $): {rmse:,.2f}")
    print("Test set size:", len(X_test), "rows | Training size:", len(X_train), "rows")
    print("==============================")

    show_equation(model)
    predict_additional(model)

    while True:
        try:
            spend = input("\nEnter an Instagram ad spend ($) to predict sales (or press Enter to quit): ")
        except EOFError:
            break
        if not spend.strip():
            break
        try:
            spend = float(spend)
        except ValueError:
            print("Please enter a number.")
            continue
        pred = predict(model, spend)
        print(f"Predicted sales for ${spend:,.2f} spend: ${pred:,.2f}")

    plot_results(df, model)


if __name__ == "__main__":
    main()