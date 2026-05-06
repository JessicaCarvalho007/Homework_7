"""
forecast_functions.py
---------------------
Helper functions shared by train_model.py and generate_forecast.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import hf_hydrodata


### Project folders ###
# Folder where this script lives:
BASE_DIR = Path(__file__).resolve().parent
# Output folders:
MODEL_PATH = BASE_DIR / "models"
FIG_PATH = BASE_DIR / "figures"
# Create folders automatically if they do not already exist:
MODEL_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH.mkdir(parents=True, exist_ok=True)


def get_training_test_data(gauge_id, train_start, train_end, test_start, test_end):
    """
    Download daily streamflow and split into train/test DataFrames.
    Both DataFrames have columns: streamflow_cfs, log_flow.
    """
    raw = hf_hydrodata.get_point_data(
        dataset="usgs_nwis",
        variable="streamflow",
        temporal_resolution="daily",
        aggregation="mean",
        site_ids=gauge_id,
        date_start=train_start,
        date_end=test_end
    )
    if 'date' in raw.columns:
        raw.index = pd.to_datetime(raw['date'])
    df = raw[[gauge_id]].rename(columns={gauge_id: 'streamflow_cfs'}).sort_index().dropna()
    df['log_flow'] = np.log(df['streamflow_cfs'] + 1)

    train = df.loc[train_start:train_end]
    test  = df.loc[test_start:test_end]
    print(f"  Training: {train.index[0].date()} to {train.index[-1].date()} ({len(train):,} days)")
    print(f"  Test:     {test.index[0].date()} to {test.index[-1].date()} ({len(test):,} days)")
    return train, test


def get_recent_data(gauge_id, forecast_date, ar_order):
    """
    Download recent observations before forecast_date and return as a DataFrame.
    Uses a metadata call to check the latest available date before downloading data.
    Raises ValueError if forecast_date is after the latest available data date,
    or if fewer than ar_order days of data precede the forecast date.
    """
    forecast_ts = pd.Timestamp(forecast_date)

    # Cheap metadata call to validate forecast_date without downloading the full record
    meta = hf_hydrodata.get_point_metadata(
                dataset="usgs_nwis",
                variable="streamflow",
                temporal_resolution="daily",
                aggregation="mean",
                site_ids=gauge_id
    )

    latest = pd.Timestamp(meta['last_date_data_available'].iloc[0])
    if forecast_ts > latest:
        raise ValueError(
            f"Forecast date {forecast_ts.date()} is after the latest available data "
            f"({latest.date()}). Choose a date on or before {latest.date()}."
        )

    # Download only the window needed: ar_order days for model seeding, 30 days for the plot
    n_days = max(ar_order, 30)
    date_start = (forecast_ts - pd.Timedelta(days=n_days)).strftime('%Y-%m-%d')
    date_end   = (forecast_ts - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    raw = hf_hydrodata.get_point_data(
                dataset="usgs_nwis",
                variable="streamflow",
                temporal_resolution="daily",
                aggregation="mean",
                site_ids=gauge_id,
                date_start=date_start,
                date_end=date_end
            )
    if 'date' in raw.columns:
        raw.index = pd.to_datetime(raw['date'])
    df = raw[[gauge_id]].rename(columns={gauge_id: 'streamflow_cfs'}).sort_index().dropna()
    df['log_flow'] = np.log(df['streamflow_cfs'] + 1)

    if len(df) < ar_order:
        raise ValueError(
            f"Need at least {ar_order} days before forecast date; only {len(df)} found."
        )
    return df


def fit_longterm_avg_model(train_df):
    """Return the mean streamflow (cfs) over the entire training period."""
    return float(train_df['streamflow_cfs'].mean())


def make_5day_forecast_longterm(mean_flow, forecast_date, n_days=5):
    """Return DataFrame with the long-term mean flow for every forecast day."""
    dates = pd.date_range(start=forecast_date, periods=n_days, freq='D')
    return pd.DataFrame({'Forecast_cfs': mean_flow}, index=dates)


def compute_metrics(observed_cfs, predicted_cfs):
    """Return dict with RMSE, R², and NSE (Nash-Sutcliffe Efficiency)."""
    obs  = np.array(observed_cfs)
    pred = np.array(predicted_cfs)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    r2   = np.corrcoef(obs, pred)[0, 1] ** 2
    nse  = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
    return {'RMSE (cfs)': rmse, 'R2': r2, 'NSE': nse}


def plot_validation(train_cfs, test_cfs, forecast_cfs, metrics, model_label, train_forecast_cfs=None, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    axes[0].plot(train_cfs.index, train_cfs.values, color='steelblue', linewidth=0.6, alpha=0.7, label='Training')
    if train_forecast_cfs is not None:
        axes[0].plot(train_forecast_cfs.index, train_forecast_cfs.values, color='tomato', linewidth=0.8, linestyle='--', alpha=0.8, label=f'{model_label} Fitted (train)')

    axes[0].plot(test_cfs.index, test_cfs.values, color='black', linewidth=1.0, label='Observed (test)')
    axes[0].plot(forecast_cfs.index, forecast_cfs.values, color='tomato', linewidth=1.2, linestyle='--', label=f'{model_label} Predicted (test)')
    axes[0].axvline(test_cfs.index[0], color='gray', linestyle=':', linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Streamflow (cfs)')
    axes[0].set_title(f'{model_label} Validation — Verde River')
    axes[0].legend(fontsize=9)

    obs  = test_cfs.values
    pred = forecast_cfs.values
    # Restrict to finite, positive pairs — multi-step AR forecasts can diverge,
    # which would otherwise collapse all visible points to the origin on a linear scale.
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs_v, pred_v = obs[valid], pred[valid]
    if len(obs_v) > 0:
        lo = min(obs_v.min(), pred_v.min())
        hi = max(obs_v.max(), pred_v.max())
        axes[1].scatter(obs_v, pred_v, alpha=0.5, color='steelblue', s=15, edgecolors='none')
        axes[1].plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='1:1 line')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')

    axes[1].set_xlabel('Observed Streamflow (cfs)')
    axes[1].set_ylabel('Predicted Streamflow (cfs)')
    axes[1].set_title(
        f"Observed vs Predicted  |  "
        f"R² = {metrics['R2']:.3f},  NSE = {metrics['NSE']:.3f},  "
        f"RMSE = {metrics['RMSE (cfs)']:.1f} cfs"
    )
    axes[1].legend()
    plt.tight_layout()
    # Save validation plot to the figures folder
    if save_path is None:
        figure_file = get_figure_file("validation_plot.png")
    else:
        figure_file = Path(save_path)

        # If user gives only a filename, save it inside FIG_PATH
        if not figure_file.is_absolute():
            figure_file = get_figure_file(figure_file.name)

    plt.savefig(figure_file, dpi=150, bbox_inches='tight')
    print(f" Plot saved to {figure_file}")
    plt.show()


def get_model_file(model_name):
    """
    Return the full path for a saved model file.

    Example:
    model_name = 'monthly_avg'
    returns: models/monthly_avg_model.pkl
    """
    return MODEL_PATH / f"{model_name}_model.pkl"


def get_figure_file(filename):
    """
    Return the full path for a figure file inside the figures folder.

    Example:
    filename = 'monthly_avg_forecast_plot.png'
    returns: figures/monthly_avg_forecast_plot.png
    """
    return FIG_PATH / filename


def save_model(model, model_name):
    """
    Save a fitted model object to the models folder.

    Parameters
    ----------
    model : object
        The fitted model or model information to save.
        Examples: float, dict, sklearn model object.
    model_name : str
        Name of the model, such as 'longterm_avg', 'monthly_avg',
        or 'weekly_regression'.
    """
    model_file = get_model_file(model_name)

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    print(f" Model saved to {model_file}")
    return model_file


def load_model(model_name):
    """
    Load a fitted model object from the models folder.

    Parameters
    ----------
    model_name : str
        Name of the model, such as 'longterm_avg', 'monthly_avg',
        or 'weekly_regression'.
    """
    model_file = get_model_file(model_name)

    if not model_file.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_file}. "
            f"Re-run the workflow with --refit True --model {model_name}."
        )

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    print(f" Model loaded from {model_file}")
    return model


def fit_monthly_avg_model(train_df):
    """
    Return a dictionary mapping each calendar month 1-12
    to the mean streamflow for that month over the training period.
    """
    return train_df.groupby(train_df.index.month)['streamflow_cfs'].mean().to_dict()


def make_5day_forecast_monthly(monthly_means, forecast_date, n_days=5):
    """
    Return a DataFrame with monthly-average forecast streamflow values.

    Each forecast day is assigned the historical mean streamflow
    for that day's calendar month.
    """
    dates = pd.date_range(start=forecast_date, periods=n_days, freq='D')
    forecasts = [monthly_means[d.month] for d in dates]

    return pd.DataFrame({'Forecast_cfs': forecasts}, index=dates)


def _add_lag_columns(df, ar_order):
    """
    Add lagged log-flow columns to a streamflow DataFrame.

    For ar_order = 7, this creates:
    log_flow_lag_1, log_flow_lag_2, ..., log_flow_lag_7

    log_flow_lag_1 is yesterday's log-flow.
    log_flow_lag_7 is the log-flow from 7 days ago.
    """
    lagged = df.copy()

    for lag in range(1, ar_order + 1):
        lagged[f'log_flow_lag_{lag}'] = lagged['log_flow'].shift(lag)

    return lagged


def fit_weekly_regression_model(train_df, ar_order=7):
    """
    Fit a simple weekly lag regression model.

    The model predicts today's log-flow using the previous ar_order days
    of log-flow values.

    For the default ar_order = 7:

        log(Q_t + 1) = b0
                       + b1 log(Q_{t-1} + 1)
                       + b2 log(Q_{t-2} + 1)
                       + ...
                       + b7 log(Q_{t-7} + 1)

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training data with columns streamflow_cfs and log_flow.

    ar_order : int
        Number of previous days used as predictors.

    Returns
    -------
    dict
        Dictionary containing the fitted regression information.
    """
    lag_cols = [f'log_flow_lag_{lag}' for lag in range(1, ar_order + 1)]

    model_df = _add_lag_columns(train_df, ar_order)
    model_df = model_df.dropna(subset=['log_flow'] + lag_cols)

    if len(model_df) == 0:
        raise ValueError(
            f"Not enough training data to fit weekly regression with ar_order={ar_order}."
        )

    y = model_df['log_flow'].values
    X = model_df[lag_cols].values

    # Add intercept column
    X_design = np.column_stack([np.ones(len(X)), X])

    # Least-squares regression using NumPy
    beta, residuals, rank, singular_values = np.linalg.lstsq(X_design, y, rcond=None)

    model = {
        'model_type': 'weekly_regression',
        'ar_order': ar_order,
        'intercept': float(beta[0]),
        'coefficients': beta[1:].tolist(),
        'feature_names': lag_cols,
        'training_rows': int(len(model_df))
    }

    return model


def make_weekly_regression_predictions(model, df):
    """
    Make one-day-ahead predictions for every day in a DataFrame.

    This is useful for validation because each prediction uses the previous
    observed streamflow values.
    """
    ar_order = int(model['ar_order'])
    intercept = float(model['intercept'])
    coefficients = np.array(model['coefficients'], dtype=float)

    lag_cols = [f'log_flow_lag_{lag}' for lag in range(1, ar_order + 1)]

    model_df = _add_lag_columns(df, ar_order)
    X = model_df[lag_cols].values

    pred_log_flow = intercept + X @ coefficients
    pred_cfs = np.maximum(np.expm1(pred_log_flow), 0)

    return pd.Series(pred_cfs, index=model_df.index, name='Forecast_cfs')


def make_5day_forecast_weekly(model, recent_df, forecast_date, n_days=5):
    """
    Generate a recursive 5-day forecast using the weekly regression model.

    Day 1 uses the most recent observed streamflow values.
    Day 2 uses the Day 1 forecast as part of its lag history.
    Day 3 uses Days 1 and 2 forecasts, and so on.
    """
    ar_order = int(model['ar_order'])
    intercept = float(model['intercept'])
    coefficients = np.array(model['coefficients'], dtype=float)

    log_history = recent_df['log_flow'].dropna().tolist()

    if len(log_history) < ar_order:
        raise ValueError(
            f"Need at least {ar_order} recent observations for weekly_regression, "
            f"but only found {len(log_history)}."
        )

    dates = pd.date_range(start=forecast_date, periods=n_days, freq='D')
    forecasts = []

    for date in dates:
        # Features are ordered as lag 1, lag 2, ..., lag ar_order
        lag_values = np.array(
            [log_history[-lag] for lag in range(1, ar_order + 1)],
            dtype=float
        )

        pred_log_flow = intercept + lag_values @ coefficients
        pred_cfs = max(np.expm1(pred_log_flow), 0)

        forecasts.append(pred_cfs)

        # Add forecast back into history for recursive multi-day prediction
        log_history.append(np.log(pred_cfs + 1))

    return pd.DataFrame({'Forecast_cfs': forecasts}, index=dates)

