from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib


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

    feature_columns = [
        "ordinal",
        "dayofweek",
        "month",
        "weekofyear",
        "sin_dayofyear",
        "cos_dayofyear",
    ]
    return df[feature_columns], df["sales"]


def train_model(csv_path: Path | str, save_path: Path | str) -> None:
    csv_path = Path(csv_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    X, y = prepare_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = Ridge(alpha=10)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = model.score(X_test, y_test)

    joblib.dump(model, save_path)

    print(f"Saved trained model to {save_path.resolve()}")
    print("Evaluation on holdout set:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")


if __name__ == "__main__":
    csv_file = Path("data/raw/sales.csv")
    model_file = Path("models/sales_forecast_model.joblib")
    train_model(csv_file, model_file)
