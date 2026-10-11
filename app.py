"""Streamlit app: predict monthly sales from a marketing budget mix.

Serves the best model saved by model_comparison.py with an interactive
dashboard: contribution breakdown, what-if analysis and historical view.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Forecast Dashboard", page_icon="📈", layout="wide")

CSV_PATH = "data/sales_data_multi.csv"
BUNDLE_PATH = "output/sales_model.joblib"
CHANNELS = {
    "Instagram ads": "instagram_ad_spend",
    "TV ads": "tv_ad_spend",
    "Email spend": "email_spend",
}
SEASONS = ["Q1", "Q2", "Q3", "Q4"]


@st.cache_resource
def load_bundle():
    bundle = joblib.load(BUNDLE_PATH)
    return bundle["model"], bundle["features"]


@st.cache_data
def load_history():
    return pd.read_csv(CSV_PATH)


def build_row(instagram, tv, email, season, features):
    row = {f: 0 for f in features}
    row["instagram_ad_spend"] = instagram
    row["tv_ad_spend"] = tv
    row["email_spend"] = email
    for q in ["Q2", "Q3", "Q4"]:
        row[f"season_{q}"] = int(season == q)
    return pd.DataFrame([row])[features]


model, features = load_bundle()
history = load_history()

st.title("📈 Sales Forecast Dashboard")
st.caption(
    "Model: **Multiple Linear Regression** (chosen by 5-fold cross-validation). "
    "Drag the sliders in the sidebar and explore contribution, what-if and history tabs."
)

with st.sidebar:
    st.header("Marketing budget")
    instagram = st.slider("Instagram ad spend ($)", 0, 10000, 2500, 100)
    tv = st.slider("TV ad spend ($)", 0, 50000, 12000, 500)
    email = st.slider("Email spend ($)", 0, 10000, 2000, 100)
    season = st.selectbox("Season", SEASONS)

row = build_row(instagram, tv, email, season, features)
prediction = float(model.predict(row)[0])
total_budget = instagram + tv + email

c1, c2, c3 = st.columns(3)
c1.metric("Predicted monthly sales", f"${prediction:,.0f}")
c2.metric("Total budget", f"${total_budget:,.0f}")
c3.metric("Return per $1 of budget", f"${prediction / total_budget:,.2f}" if total_budget > 0 else "—")

tab_contrib, tab_whatif, tab_history, tab_planner = st.tabs(
    ["🔍 Contribution", "🎯 What-if analysis", "📅 Historical view", "🗓️ Budget planner"]
)

with tab_contrib:
    st.subheader("Where the predicted sales come from")
    coefs = pd.Series(model.coef_, index=features)
    coefs.index = [
        "Instagram ads" if c == "instagram_ad_spend"
        else "TV ads" if c == "tv_ad_spend"
        else "Email spend" if c == "email_spend"
        else c.replace("season_", "Seasonality: ")
        for c in coefs.index
    ]
    contributions = pd.Series(
        {
            "Instagram ads": coefs["Instagram ads"] * instagram,
            "TV ads": coefs["TV ads"] * tv,
            "Email spend": coefs["Email spend"] * email,
            "Seasonality": coefs[f"Seasonality: {season}"] if season != "Q1" else 0,
            "Baseline (intercept)": model.intercept_,
        }
    )
    contributions = contributions.round(0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(contributions.index, contributions.values, color="#1f77b4")
    ax.set_xlabel("Contribution to monthly sales ($)")
    ax.set_title("Sales breakdown by budget line")
    ax.grid(axis="x", alpha=0.3)
    for i, (v) in enumerate(contributions.values):
        ax.annotate(f"${v:,.0f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center")
    st.pyplot(fig, clear_figure=True)
    st.caption("Bar width = coefficient × spend for that channel. Q1 is the reference season (no seasonality effect).")

with tab_whatif:
    st.subheader("What happens if I change one channel?")
    channel = st.selectbox("Channel to scale", list(CHANNELS.keys()))
    column = CHANNELS[channel]
    scale = st.slider(f"New budget for {channel} ($)", 0, 60000, 15000, 500)

    def predict_with(sub, ch):
        level = CHANNELS[ch]
        inputs = {
            "instagram_ad_spend": instagram, "tv_ad_spend": tv, "email_spend": email
        }
        inputs[level] = sub
        return float(model.predict(build_row(inputs["instagram_ad_spend"], inputs["tv_ad_spend"], inputs["email_spend"], season, features))[0])

    current_other = {
        chs: (instagram if c == "instagram_ad_spend" else tv if c == "tv_ad_spend" else email)
        for c, chs in CHANNELS.items()
    }.get(channel, 0)
    current_pred = predict_with(current_other, channel)
    new_pred = predict_with(scale, channel)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current prediction", f"${current_pred:,.0f}")
    col2.metric("New prediction", f"${new_pred:,.0f}")
    col3.metric("Change in sales", f"${new_pred - current_pred:+,.0f}")

    st.markdown("**Incremental sales at selected budgets:**")
    levels = np.linspace(0, scale, 6, dtype=int)
    whatif = pd.DataFrame({"Budget ($)": levels})
    whatif["Predicted sales ($)"] = [predict_with(v, channel) for v in levels]
    whatif["Δ vs current ($)"] = whatif["Predicted sales ($)"] - current_pred
    st.dataframe(whatif, use_container_width=True, hide_index=True)

with tab_history:
    st.subheader("Historical monthly sales vs budget (last 24 months)")
    plot_df = history.copy()
    plot_df["total_budget"] = (
        plot_df["instagram_ad_spend"] + plot_df["tv_ad_spend"] + plot_df["email_spend"]
    )
    recent = plot_df.tail(24)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent["month"], recent["sales"], label="Actual sales", marker="o", markersize=3)
    ax.plot(recent["month"], recent["total_budget"], label="Total budget", marker="o", markersize=3)
    ax.set_ylabel("$")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    st.caption("Recent rows used for training (96 months total):")
    st.dataframe(history.tail(12), use_container_width=True, hide_index=True)

with tab_planner:
    st.subheader("How much budget do I need to hit my target?")
    st.caption(
        "Your current budget mix (IG / TV / Email) is scaled up or down in the "
        "same proportions to reach a target. Because seasons carry a free sales "
        "lift, the needed budget differs by season."
    )

    target_mode = st.radio(
        "Target sales",
        ["Current predicted sales", "Custom value"],
        horizontal=True,
    )
    if target_mode == "Custom value":
        target = st.number_input("Target monthly sales ($)", min_value=0, value=int(prediction), step=25000)
    else:
        target = prediction

    if total_budget <= 0:
        st.warning("Set at least one channel budget above $0 to use the planner.")
    else:
        coef_row = pd.Series(model.coef_, index=features)
        channel_contrib = (
            coef_row["instagram_ad_spend"] * instagram
            + coef_row["tv_ad_spend"] * tv
            + coef_row["email_spend"] * email
        )

        rows = []
        for s in SEASONS:
            season_effect = coef_row[f"season_{s}"] if s != "Q1" else 0.0
            needed_scale = (
                (target - model.intercept_ - season_effect) / channel_contrib
                if abs(channel_contrib) > 1e-9
                else float("nan")
            )
            needed_scale = float(np.clip(needed_scale, 0, 5))
            scaled_budget = int(round((instagram * needed_scale + tv * needed_scale + email * needed_scale) / 1000) * 1000)
            check_budget = (instagram * needed_scale, tv * needed_scale, email * needed_scale)
            achieved = float(
                model.predict(
                    build_row(check_budget[0], check_budget[1], check_budget[2], s, features)
                )[0]
            )
            rows.append(
                {
                    "Season": s,
                    "Budget scale": f"{needed_scale:.2f}x",
                    "Total budget needed ($)": scaled_budget,
                    "Predicted sales ($)": int(round(achieved)),
                }
            )

        planner = pd.DataFrame(rows)
        st.dataframe(planner, use_container_width=True, hide_index=True)

        q1_cost = planner.loc[planner["Season"] == "Q1", "Total budget needed ($)"].iloc[0]
        q4_cost = planner.loc[planner["Season"] == "Q4", "Total budget needed ($)"].iloc[0]
        if q1_cost > 0:
            delta = q4_cost - q1_cost
            sign = "more" if delta > 0 else "less"
            st.info(
                f"Q4 needs **${abs(delta):,} {sign}** budget than Q1 for the same "
                f"${target:,.0f} target — the holiday season does part of the work for you."
            )

st.divider()
st.caption("Model: " + type(model).__name__ + " · Explorer: contributions from linear coefficients.")