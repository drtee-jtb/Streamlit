from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sales_forecast_model.joblib"
DATA_PATH = ROOT_DIR / "data" / "raw" / "sales.csv"
FEATURE_COLUMNS = [
    "ordinal",
    "dayofweek",
    "month",
    "weekofyear",
    "sin_dayofyear",
    "cos_dayofyear",
]


@st.cache_data
def load_sales_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sales data not found: {path.resolve()}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource
def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path.resolve()}")
    return joblib.load(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["ordinal"] = df["date"].map(pd.Timestamp.toordinal)

    return df[FEATURE_COLUMNS], df["sales"]


def make_feature_frame(dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)
    frame = pd.DataFrame(
        {
            "ordinal": dates.map(pd.Timestamp.toordinal),
            "dayofweek": dates.dt.dayofweek,
            "month": dates.dt.month,
            "weekofyear": dates.dt.isocalendar().week.astype(int),
            "dayofyear": dates.dt.dayofyear,
        }
    )
    frame["sin_dayofyear"] = np.sin(2 * np.pi * frame["dayofyear"] / 365.25)
    frame["cos_dayofyear"] = np.cos(2 * np.pi * frame["dayofyear"] / 365.25)
    return frame[FEATURE_COLUMNS]


def forecast_sales(model, target_dates: list[date]) -> pd.DataFrame:
    features = make_feature_frame(pd.Series(target_dates))
    predictions = model.predict(features)
    forecast_df = pd.DataFrame({"date": target_dates, "predicted_sales": predictions})
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    return forecast_df


def evaluate_model(model, data: pd.DataFrame) -> dict[str, float]:
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = model.score(X_test, y_test)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def render_sidebar(last_date: date) -> tuple[str, date, int]:
    st.sidebar.header("Forecast settings")
    mode = st.sidebar.selectbox(
        "Forecast mode",
        ["Single date prediction", "Multi-day forecast"],
    )
    start_date = st.sidebar.date_input(
        "Forecast start date",
        value=last_date + timedelta(days=1),
        min_value=last_date + timedelta(days=1),
    )
    days = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)
    return mode, start_date, days


def format_label(value: float) -> str:
    return f"{value:,.2f}"


def main() -> None:
    st.set_page_config(page_title="Sales Forecast App", layout="wide")

    st.title(":bar_chart: Sales Forecast App")
    st.markdown(
        "Use the trained Ridge regression model to forecast future sales from historical data. "
        "Adjust the forecast horizon and compare predicted values against the actual sales history."
    )

    data = load_sales_data()
    model = load_model()
    last_date = data["date"].max().date()
    model_metrics = evaluate_model(model, data)

    mode, start_date, days = render_sidebar(last_date)

    summary_col, details_col = st.columns([2, 1])
    summary_col.subheader("Dataset overview")
    summary_col.metric("Number of records", len(data))
    summary_col.metric("Historical start", data["date"].min().strftime("%Y-%m-%d"))
    summary_col.metric("Historical end", last_date.strftime("%Y-%m-%d"))
    summary_col.write(
        "This model was trained using date-derived features and the historical sales column from `data/raw/sales.csv`."
    )

    details_col.subheader("Model evaluation")
    details_col.metric("MAE", format_label(model_metrics["MAE"]))
    details_col.metric("RMSE", format_label(model_metrics["RMSE"]))
    details_col.metric("R²", f"{model_metrics['R2']:.3f}")
    details_col.caption("Evaluation is computed on the last 20% of the historical dataset.")

    st.markdown("---")

    if mode == "Single date prediction":
        prediction_df = forecast_sales(model, [start_date])
        predicted_value = float(prediction_df["predicted_sales"].iloc[0])

        st.subheader("Single Date Prediction")
        col1, col2 = st.columns([1, 2])
        col1.metric("Selected date", start_date.strftime("%Y-%m-%d"))
        col1.metric("Predicted sales", format_label(predicted_value))
        col2.write(
            prediction_df.assign(date=prediction_df["date"].dt.strftime("%Y-%m-%d"))
        )
    else:
        target_dates = [start_date + timedelta(days=i) for i in range(days)]
        prediction_df = forecast_sales(model, target_dates)

        st.subheader("Forecast results")
        st.write(
            "Predicted sales for the selected horizon. Use the chart below to compare the forecast direction."
        )
        st.dataframe(
            prediction_df.assign(date=prediction_df["date"].dt.strftime("%Y-%m-%d")),
            use_container_width=True,
        )
        st.line_chart(
            prediction_df.set_index("date")["predicted_sales"],
            height=360,
        )

    st.markdown("---")
    st.subheader("Historical sales")
    hist_col, chart_col = st.columns([1, 2])
    hist_col.write(
        "The chart below shows the complete sales history used by the model. "
        "A clean date index and engineered time features are used to learn patterns."
    )
    hist_col.dataframe(
        data.tail(10).assign(date=data["date"].dt.strftime("%Y-%m-%d")),
        use_container_width=True,
    )
    chart_col.line_chart(
        data.set_index("date")["sales"].rename("actual_sales"),
        height=420,
    )

    if mode == "Multi-day forecast":
        forecast_data = prediction_df.copy()
        forecast_data["date"] = pd.to_datetime(forecast_data["date"])
        combined = pd.concat(
            [
                data.set_index("date")["sales"].rename("historical"),
                forecast_data.set_index("date")["predicted_sales"].rename("forecast"),
            ],
            axis=1,
        )
        st.markdown("---")
        st.subheader("Forecast vs Historical")
        st.line_chart(combined, height=420)

    st.markdown(
        "---\n"
        "Built with Streamlit. The model is loaded from `models/sales_forecast_model.joblib`. "
        "If you want to retrain, use `python src/train_model.py` from the project root."
    )

    with st.expander("Raw sales data preview"):
        st.dataframe(data.assign(date=data["date"].dt.strftime("%Y-%m-%d")))


if __name__ == "__main__":
    main()
