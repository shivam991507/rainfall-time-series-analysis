"""
Rainfall Time Series Analysis - Delhi, Haryana & Chandigarh (1966-2015)

This script was written from scratch in Python after reviewing the original
academic report and the supplied spreadsheet/CSV. It intentionally uses only
the raw monthly date and rainfall columns from the source file; the many
precomputed spreadsheet columns are ignored.

Analyses reproduced:
1. Monthly time plot
2. Annual average rainfall and linear trend
3. Multiplicative decomposition: Trend x Seasonal x Cyclic x Random
4. Seasonal indices and deseasonalized series
5. Harmonic analysis of 20 trial periods for the cyclic component
6. Variate-difference table
7. ADF stationarity test
8. ACF/PACF diagnostics
9. Model comparison:
   - Simple Seasonal exponential smoothing
   - SARIMA(0,0,0)(0,1,1)[12]
   - ARIMA(2,0,2)
10. Forecast from 2016 through 2025 using the best model

The original report covers two regions. The supplied CSV contains the monthly
series for Delhi, Haryana & Chandigarh only. The same functions can be reused
for the Assam-Meghalaya series once its raw monthly data are supplied.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, f as f_dist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "delhi_haryana_chandigarh_source.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

for directory in [PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_monthly_series(path: Path) -> pd.Series:
    """Extract the monthly date and rainfall columns from the supplied messy CSV."""
    raw = pd.read_csv(path, header=None)

    header_matches = raw.apply(
        lambda col: col.astype(str).str.contains("Year - Month", case=False, na=False)
    )
    rows, cols = np.where(header_matches.to_numpy())
    if len(rows) == 0:
        raise ValueError("Could not find the 'Year - Month' header in the CSV.")

    header_row = int(rows[0])
    date_col = int(cols[0])
    rainfall_col = date_col + 1

    df = raw.iloc[header_row + 1 :, [date_col, rainfall_col]].copy()
    df.columns = ["year_month", "rainfall"]

    valid = df["year_month"].astype(str).str.match(r"^\s*\d{4}\s*-\s*[A-Za-z]{3}\s*$")
    df = df.loc[valid].copy()

    cleaned_dates = (
        df["year_month"].astype(str)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    df["date"] = pd.to_datetime(cleaned_dates, format="%Y-%b", errors="raise")
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")

    # November 1970 is blank in the supplied file. Filling it with 0.0
    # reproduces the 1970 annual average (44.85) reported in the original work.
    if df["rainfall"].isna().any():
        missing_dates = df.loc[df["rainfall"].isna(), "date"].dt.strftime("%Y-%b").tolist()
        print(f"Missing rainfall values found at {missing_dates}; filling with 0.0.")
        df["rainfall"] = df["rainfall"].fillna(0.0)

    series = (
        df.set_index("date")["rainfall"]
        .astype(float)
        .sort_index()
        .asfreq("MS")
    )

    if series.isna().any():
        raise ValueError("Unexpected gaps remain after converting to monthly frequency.")
    return series


def fit_linear_trend_from_annual_means(series: pd.Series):
    """Fit a linear trend to annual-average rainfall on the monthly time scale."""
    annual = series.resample("YS").mean()
    repeated_annual = np.repeat(annual.to_numpy(), 12)
    t = np.arange(1, len(series) + 1, dtype=float)

    trend_fit = linregress(t, repeated_annual)
    trend = pd.Series(
        trend_fit.intercept + trend_fit.slope * t,
        index=series.index,
        name="trend"
    )
    return annual, trend, trend_fit


def multiplicative_components(series: pd.Series, trend: pd.Series):
    """
    Estimate seasonal, cyclic and random components on a percentage scale.

    Y_t = T_t * (S_t/100) * (C_t/100) * (R_t/100)
    """
    detrended = 100.0 * series / trend

    seasonal_index = detrended.groupby(detrended.index.month).mean()
    seasonal_index = seasonal_index / seasonal_index.mean() * 100.0

    seasonal = pd.Series(
        [seasonal_index.loc[m] for m in series.index.month],
        index=series.index,
        name="seasonal_index"
    )

    deseasonalized = 100.0 * series / seasonal
    cycle_irregular = 100.0 * series / (trend * (seasonal / 100.0))

    return detrended, seasonal_index, seasonal, deseasonalized, cycle_irregular


def harmonic_analysis(cycle_irregular: pd.Series, n_trials: int = 20):
    """
    Harmonic analysis using 20 trial periods.

    Trial periods are n/k, k=1,...,20. For each trial period T:
        z_t = a0 + a cos(2*pi*t/T) + b sin(2*pi*t/T) + error
    """
    z = cycle_irregular.to_numpy(dtype=float)
    n = len(z)
    t = np.arange(1, n + 1, dtype=float)
    periods = n / np.arange(1, n_trials + 1, dtype=float)

    rows = []
    tss = np.sum((z - z.mean()) ** 2)

    for T in periods:
        X = np.column_stack([
            np.ones(n),
            np.cos(2 * np.pi * t / T),
            np.sin(2 * np.pi * t / T),
        ])
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        fitted = X @ beta

        ss_reg = np.sum((fitted - z.mean()) ** 2)
        ss_err = np.sum((z - fitted) ** 2)
        df_reg = 2
        df_err = n - 3
        f_stat = (ss_reg / df_reg) / (ss_err / df_err)
        p_value = f_dist.sf(f_stat, df_reg, df_err)

        rows.append({
            "trial_period_months": T,
            "trial_period_years": T / 12.0,
            "a0": beta[0],
            "cos_coef": beta[1],
            "sin_coef": beta[2],
            "explained_ss": ss_reg,
            "total_ss": tss,
            "f_stat": f_stat,
            "p_value": p_value,
        })

    table = pd.DataFrame(rows).sort_values("explained_ss", ascending=False).reset_index(drop=True)
    best = table.iloc[0]

    T = float(best["trial_period_months"])
    X_best = np.column_stack([
        np.ones(n),
        np.cos(2 * np.pi * t / T),
        np.sin(2 * np.pi * t / T),
    ])
    beta = np.array([best["a0"], best["cos_coef"], best["sin_coef"]], dtype=float)
    cyclic = pd.Series(X_best @ beta, index=cycle_irregular.index, name="cyclic_index")
    return table, cyclic


def variate_difference_table(series: pd.Series):
    """Create annual-average, first-difference and second-difference tables."""
    annual = series.resample("YS").mean()
    out = pd.DataFrame({
        "year": annual.index.year,
        "annual_average": annual.to_numpy()
    })
    out["first_difference"] = out["annual_average"].diff()
    out["second_difference"] = out["first_difference"].diff()

    m0 = np.mean(out["annual_average"].to_numpy() ** 2)
    d1 = out["first_difference"].dropna().to_numpy()
    d2 = out["second_difference"].dropna().to_numpy()
    m1 = np.mean(d1 ** 2)
    m2 = np.mean(d2 ** 2)

    summary = pd.DataFrame({
        "order": [0, 1, 2],
        "second_moment": [m0, m1, m2],
        "variance_estimate": [m0, m1 / 2.0, m2 / 6.0],
    })
    return out, summary


def stationarity_and_correlogram(series: pd.Series):
    """
    Use lag 8 and constant + linear trend so the ADF setup matches the
    original report's reported test closely.
    """
    adf_result = adfuller(series, maxlag=8, autolag=None, regression="ct")
    adf_summary = pd.DataFrame([{
        "adf_statistic": adf_result[0],
        "p_value": adf_result[1],
        "lag_order": adf_result[2],
        "n_observations": adf_result[3],
        "stationary_at_5pct": adf_result[1] < 0.05,
    }])

    acf_vals = acf(series, nlags=24, fft=True)
    pacf_vals = pacf(series, nlags=24, method="ywm")
    corr = pd.DataFrame({
        "lag": np.arange(25),
        "acf": acf_vals,
        "pacf": pacf_vals,
    })
    return adf_summary, corr


def regression_metrics(actual, fitted, start=0):
    actual = np.asarray(actual, dtype=float)[start:]
    fitted = np.asarray(fitted, dtype=float)[start:]
    mask = np.isfinite(actual) & np.isfinite(fitted)
    actual = actual[mask]
    fitted = fitted[mask]
    return {
        "r_squared": r2_score(actual, fitted),
        "rmse": np.sqrt(mean_squared_error(actual, fitted)),
        "mae": mean_absolute_error(actual, fitted),
    }


def fit_models(series: pd.Series):
    """
    Fit the three Delhi/Haryana/Chandigarh models compared in the report.
    """
    simple_seasonal = ExponentialSmoothing(
        series,
        trend=None,
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)

    sarima = SARIMAX(
        series,
        order=(0, 0, 0),
        seasonal_order=(0, 1, 1, 12),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    arima = ARIMA(
        series,
        order=(2, 0, 2),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()

    rows = []

    m = regression_metrics(series, simple_seasonal.fittedvalues, start=0)
    rows.append({"model": "Simple Seasonal", **m})

    m = regression_metrics(series, sarima.fittedvalues, start=12)
    rows.append({"model": "SARIMA(0,0,0)(0,1,1)[12]", **m})

    m = regression_metrics(series, arima.fittedvalues, start=0)
    rows.append({"model": "ARIMA(2,0,2)", **m})

    comparison = pd.DataFrame(rows)
    comparison["rank_rmse"] = comparison["rmse"].rank(method="min")
    comparison = comparison.sort_values("rmse").reset_index(drop=True)
    return simple_seasonal, sarima, arima, comparison


def save_figures(series, annual, trend, seasonal_index, deseasonalized,
                 harmonic_table, cyclic, random_component, corr, forecast):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values)
    ax.set_title("Monthly Rainfall - Delhi, Haryana & Chandigarh (1966-2015)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_monthly_time_plot.png", dpi=160)
    plt.close(fig)

    annual_trend = trend.resample("YS").mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(annual.index.year, annual.values, marker="o", label="Annual average")
    ax.plot(annual_trend.index.year, annual_trend.values, label="Linear trend")
    ax.set_title("Annual Average Rainfall and Linear Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average monthly rainfall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_annual_trend.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, 13), seasonal_index.values, marker="o")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_title("Seasonal Indices")
    ax.set_xlabel("Month")
    ax.set_ylabel("Seasonal index (mean = 100)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_seasonal_indices.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(deseasonalized.index, deseasonalized.values)
    ax.set_title("Deseasonalized Rainfall Series")
    ax.set_xlabel("Year")
    ax.set_ylabel("Deseasonalized rainfall")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_deseasonalized.png", dpi=160)
    plt.close(fig)

    hp = harmonic_table.sort_values("trial_period_months")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hp["trial_period_months"], hp["explained_ss"], marker="o")
    ax.set_title("Harmonic Analysis - 20 Trial Periods")
    ax.set_xlabel("Trial period (months)")
    ax.set_ylabel("Explained sum of squares")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_harmonic_trial_periods.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cyclic.index, cyclic.values)
    ax.set_title("Estimated Cyclic Component")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cyclic index")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_cyclic_component.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(random_component.index, random_component.values)
    ax.set_title("Estimated Random Component")
    ax.set_xlabel("Year")
    ax.set_ylabel("Random index")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_random_component.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(corr["lag"], corr["acf"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Autocorrelation Function")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Autocorrelation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_acf.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    observed_tail = series.loc["2006-01-01":]
    ax.plot(observed_tail.index, observed_tail.values, label="Observed")
    ax.plot(forecast.index, forecast.values, label="Forecast")
    ax.set_title("Rainfall Forecast: 2016-2025")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_forecast_2016_2025.png", dpi=160)
    plt.close(fig)


def main():
    series = load_monthly_series(RAW_FILE)
    series.rename("rainfall").to_csv(PROCESSED_DIR / "monthly_rainfall_1966_2015.csv")

    annual, trend, trend_fit = fit_linear_trend_from_annual_means(series)
    detrended, seasonal_index, seasonal, deseasonalized, cycle_irregular = (
        multiplicative_components(series, trend)
    )

    harmonic_table, cyclic = harmonic_analysis(cycle_irregular, n_trials=20)
    random_component = 100.0 * cycle_irregular / cyclic

    components = pd.DataFrame({
        "rainfall": series,
        "trend": trend,
        "seasonal_index": seasonal,
        "deseasonalized": deseasonalized,
        "cycle_irregular_index": cycle_irregular,
        "cyclic_index": cyclic,
        "random_index": random_component,
    })
    components.to_csv(PROCESSED_DIR / "decomposition_components.csv")

    annual_table, variate_summary = variate_difference_table(series)
    annual_table.to_csv(RESULTS_DIR / "variate_difference_table.csv", index=False)
    variate_summary.to_csv(RESULTS_DIR / "variate_difference_summary.csv", index=False)

    pd.DataFrame({
        "month": ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"],
        "seasonal_index": seasonal_index.values,
    }).to_csv(RESULTS_DIR / "seasonal_indices.csv", index=False)

    harmonic_table.to_csv(RESULTS_DIR / "harmonic_trial_periods.csv", index=False)

    adf_summary, corr = stationarity_and_correlogram(series)
    adf_summary.to_csv(RESULTS_DIR / "adf_test.csv", index=False)
    corr.to_csv(RESULTS_DIR / "correlogram.csv", index=False)

    simple_seasonal, sarima, arima, model_comparison = fit_models(series)
    model_comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    forecast = simple_seasonal.forecast(120)
    forecast.name = "forecast"
    forecast.to_csv(RESULTS_DIR / "forecast_2016_2025.csv")

    model_parameters = {
        "trend_slope_per_month": trend_fit.slope,
        "trend_intercept": trend_fit.intercept,
        "best_harmonic_period_months": float(harmonic_table.iloc[0]["trial_period_months"]),
        "best_harmonic_period_years": float(harmonic_table.iloc[0]["trial_period_years"]),
        "simple_seasonal_smoothing_level": float(simple_seasonal.params["smoothing_level"]),
        "simple_seasonal_smoothing_seasonal": float(simple_seasonal.params["smoothing_seasonal"]),
        "sarima_seasonal_ma_L12": float(sarima.params.get("ma.S.L12", np.nan)),
        "arima_constant": float(arima.params.get("const", np.nan)),
        "arima_ar_L1": float(arima.params.get("ar.L1", np.nan)),
        "arima_ar_L2": float(arima.params.get("ar.L2", np.nan)),
        "arima_ma_L1": float(arima.params.get("ma.L1", np.nan)),
        "arima_ma_L2": float(arima.params.get("ma.L2", np.nan)),
    }
    pd.DataFrame([model_parameters]).to_csv(
        RESULTS_DIR / "model_parameters.csv", index=False
    )

    save_figures(
        series, annual, trend, seasonal_index, deseasonalized,
        harmonic_table, cyclic, random_component, corr, forecast
    )

    print("\n--- Dataset ---")
    print(f"Observations: {len(series)} monthly values")
    print(f"Period: {series.index.min():%Y-%m} to {series.index.max():%Y-%m}")
    print(f"Maximum rainfall: {series.max():.1f}")

    print("\n--- Linear trend ---")
    print(f"Monthly slope: {trend_fit.slope:.6f}")
    print(f"Approx. annual slope: {12 * trend_fit.slope:.3f}")

    print("\n--- Seasonality ---")
    print(pd.DataFrame({
        "month": ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"],
        "index": seasonal_index.values,
    }).to_string(index=False))

    print("\n--- Harmonic analysis ---")
    print(harmonic_table.head(5)[
        ["trial_period_months", "trial_period_years", "f_stat", "p_value"]
    ].to_string(index=False))
    print(
        f"Selected dominant cycle: "
        f"{harmonic_table.iloc[0]['trial_period_months']:.1f} months "
        f"({harmonic_table.iloc[0]['trial_period_years']:.2f} years)"
    )

    print("\n--- ADF test ---")
    print(adf_summary.to_string(index=False))

    print("\n--- Model comparison ---")
    print(model_comparison.to_string(index=False))

    print("\nBest model by RMSE:", model_comparison.iloc[0]["model"])
    print("Forecast saved for 2016-2025.")


if __name__ == "__main__":
    main()
