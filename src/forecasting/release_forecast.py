import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import sys
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import VIZ_DIR, REPORTS_DIR, FORECAST_HORIZON
from src.utils.helpers import save_json

warnings.filterwarnings("ignore")

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    pass


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    if "year_added" in df.columns:
        year_col = "year_added"
    elif "release_year" in df.columns:
        year_col = "release_year"
    else:
        raise ValueError("No year column found")

    monthly = df.copy()
    if "date_added" in df.columns and df["date_added"].notna().any():
        monthly["date"] = pd.to_datetime(df["date_added"], errors="coerce")
    else:
        monthly["date"] = pd.to_datetime(df[year_col].astype(str) + "-01-01")
    monthly = monthly.dropna(subset=["date"])
    monthly = monthly.set_index("date").resample("ME").size().to_frame("count")
    monthly.index = monthly.index.to_period("M").to_timestamp()
    return monthly


def train_test_split_ts(series: pd.Series, test_months: int = FORECAST_HORIZON):
    train = series[:-test_months]
    test = series[-test_months:]
    return train, test


def evaluate_forecast(y_true, y_pred, model_name: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return {"model": model_name, "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


def run_arima(train: pd.Series, test: pd.Series, order=(1, 1, 1)):
    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        return evaluate_forecast(test.values, forecast.values, "ARIMA"), forecast.values
    except Exception as e:
        print(f"  ARIMA failed: {e}")
        return {"model": "ARIMA", "mae": None, "rmse": None, "mape": None}, None


def run_arima_auto(train: pd.Series, test: pd.Series):
    best_aic = np.inf
    best_order = None
    best_forecast = None
    p_range = range(0, 4)
    d_range = range(0, 2)
    q_range = range(0, 4)
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(train, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
                best_forecast = fitted.forecast(steps=len(test)).values
        except Exception:
            continue
    if best_order:
        print(f"  Best ARIMA order: {best_order} (AIC={best_aic:.1f})")
        metrics = evaluate_forecast(test.values, best_forecast, "ARIMA (auto)")
        return metrics, best_forecast
    return {"model": "ARIMA (auto)", "mae": None, "rmse": None, "mape": None}, None


def run_exponential_smoothing(train: pd.Series, test: pd.Series):
    try:
        model = ExponentialSmoothing(train, seasonal_periods=12, trend="add", seasonal="add")
        fitted = model.fit()
        forecast = fitted.forecast(len(test))
        return evaluate_forecast(test.values, forecast.values, "ExpSmoothing"), forecast.values
    except Exception as e:
        print(f"  ExpSmoothing failed: {e}")
        return {"model": "ExpSmoothing", "mae": None, "rmse": None, "mape": None}, None


def run_prophet(train: pd.Series, test: pd.Series):
    if not PROPHET_AVAILABLE:
        return {"model": "Prophet", "mae": None, "rmse": None, "mape": None}, None
    try:
        df_prophet = pd.DataFrame({
            "ds": train.index,
            "y": train.values
        })
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=len(test), freq="ME")
        forecast = model.predict(future)
        forecast_values = forecast["yhat"].values[-len(test):]
        metrics = evaluate_forecast(test.values, forecast_values, "Prophet")
        return metrics, forecast_values
    except Exception as e:
        print(f"  Prophet failed: {e}")
        return {"model": "Prophet", "mae": None, "rmse": None, "mape": None}, None


def run_forecast_pipeline(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("TASK 5: FORECAST NETFLIX RELEASE TRENDS")
    print("=" * 60)

    ts_data = prepare_time_series(df)
    if "year_added" in df.columns and df["year_added"].notna().any():
        print(f"  Using 'year_added' time series")
    else:
        print(f"  Using 'release_year' time series")

    print(f"  Time series length: {len(ts_data)} months")
    print(f"  Date range: {ts_data.index[0].date()} to {ts_data.index[-1].date()}")
    print(f"  Total releases: {int(ts_data['count'].sum())}")
    print(f"  Avg monthly releases: {ts_data['count'].mean():.1f}")

    train, test = train_test_split_ts(ts_data["count"], test_months=FORECAST_HORIZON)
    print(f"  Train: {len(train)} months, Test: {len(test)} months")

    fig, ax = plt.subplots(figsize=(14, 5))
    train.plot(ax=ax, label="Train", color="blue")
    test.plot(ax=ax, label="Test", color="green")
    ax.set_title("Monthly Content Additions to Netflix")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "time_series.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = []
    forecasts = {}

    arima_metrics, arima_forecast = run_arima_auto(train, test)
    results.append(arima_metrics)
    forecasts["ARIMA"] = arima_forecast

    es_metrics, es_forecast = run_exponential_smoothing(train, test)
    results.append(es_metrics)
    forecasts["ExpSmoothing"] = es_forecast

    prophet_metrics, prophet_forecast = run_prophet(train, test)
    results.append(prophet_metrics)
    forecasts["Prophet"] = prophet_forecast

    results_df = pd.DataFrame(results)
    best_row = results_df.dropna(subset=["mae"]).sort_values("mae").iloc[0] if not results_df.dropna(subset=["mae"]).empty else None

    print(f"\n--- Forecast Comparison ---")
    print(results_df.to_string(index=False))

    if best_row is not None:
        print(f"\n  Best model: {best_row['model']} (MAE={best_row['mae']})")

    plot_forecasts(train, test, forecasts)
    save_json(results_df.to_dict(orient="records"), REPORTS_DIR / "forecast_results.json")

    return results_df, forecasts, ts_data


def plot_forecasts(train, test, forecasts):
    fig, ax = plt.subplots(figsize=(14, 6))
    train.plot(ax=ax, label="Train", color="blue", linewidth=2)
    test.plot(ax=ax, label="Actual", color="green", linewidth=2)
    colors = {"ARIMA": "red", "ExpSmoothing": "orange", "Prophet": "purple"}
    for name, forecast in forecasts.items():
        if forecast is not None:
            ax.plot(test.index, forecast, label=name, linestyle="--", color=colors.get(name, "gray"), alpha=0.7)
    ax.set_title("Forecast Comparison")
    ax.set_ylabel("Monthly Additions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "forecast_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
