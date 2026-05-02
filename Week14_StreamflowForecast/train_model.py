"""
train_model.py
--------------
Fits the chosen model on the training period and optionally validates on the
test period. Run via run_workflow.sh, or directly:

    python train_model.py --email YOU@EMAIL.COM --pin 1234 [--other-options]
"""

import argparse
import pandas as pd
import hf_hydrodata
from forecast_functions import (
    get_training_test_data,
    fit_monthly_avg_model,
    fit_longterm_avg_model,
    fit_weekly_regression_model,
    make_weekly_regression_predictions,
    compute_metrics,
    plot_validation,
    save_model,
    load_model,
    get_model_file,
)

parser = argparse.ArgumentParser()
parser.add_argument('--email',       required=True)
parser.add_argument('--pin',         required=True)
parser.add_argument('--gauge-id',    default='09506000')
parser.add_argument('--ar-order',    type=int, default=7)
parser.add_argument('--train-start', default='1990-01-01')
parser.add_argument('--train-end',   default='2022-12-31')
parser.add_argument('--test-start',  default='2023-01-01')
parser.add_argument('--test-end',    default='2024-12-31')
parser.add_argument('--model', default='longterm_avg', choices=['longterm_avg', 'monthly_avg', 'weekly_regression'])
parser.add_argument('--refit',       default='True')
parser.add_argument('--validate',    default='True')
args = parser.parse_args()

REFIT_MODEL    = args.refit.lower()    == 'true'
RUN_VALIDATION = args.validate.lower() == 'true'

hf_hydrodata.register_api_pin(email=args.email, pin=args.pin)

print("\n--- Step 1: Download streamflow data ---")
train, test = get_training_test_data(args.gauge_id, args.train_start, args.train_end, args.test_start, args.test_end)

# ── Long-term average model ───────────────────────────────────────────────────
if args.model == 'longterm_avg':
    print("\n--- Step 2: Fit long-term average model ---")
    model_file = get_model_file(args.model)

    if REFIT_MODEL or not model_file.exists():
        mean_flow = fit_longterm_avg_model(train)
        print(f" Long-term mean: {mean_flow:.2f} cfs")
        save_model(mean_flow, args.model)
    else:
        mean_flow = load_model(args.model)
        if not isinstance(mean_flow, float):
            raise TypeError(
                "saved_model.pkl does not contain a longterm_avg model. "
                "Re-run with --refit True --model longterm_avg to train one."
            )

    if RUN_VALIDATION:
        print("\n--- Step 3: Validate on test period ---")
        train_fitted    = pd.Series(mean_flow, index=train.index)
        forecast_series = pd.Series(mean_flow, index=test.index)

        metrics = compute_metrics(test['streamflow_cfs'].values, forecast_series.values)
        print("\n  Validation metrics:")
        for name, val in metrics.items():
            print(f"    {name:<12}: {val:.4f}")
        print("\n  NSE guide: >0.75 very good | 0.65–0.75 good | "
              "0.50–0.65 satisfactory | <0.50 poor")

        plot_validation(
            train['streamflow_cfs'],
            test['streamflow_cfs'],
            forecast_series,
            metrics,
            'Long-term Average',
            train_forecast_cfs=train_fitted,
            save_path=f"{args.model}_validation_plot.png"
        )

elif args.model == 'monthly_avg':

    print("\n--- Step 2: Fit/load monthly average model ---")

    model_file = get_model_file(args.model)

    if REFIT_MODEL or not model_file.exists():
        monthly_means = fit_monthly_avg_model(train)
        save_model(monthly_means, args.model)
    else:
        monthly_means = load_model(args.model)

    if not isinstance(monthly_means, dict):
        raise TypeError(
            f"{model_file} does not contain a monthly_avg model. "
            "Re-run with --refit True --model monthly_avg to train one."
        )

    model_label = 'Monthly Average'

    if RUN_VALIDATION:

        print("\n--- Step 3: Validate monthly average model ---")

        train_fitted = pd.Series(
            [monthly_means[d.month] for d in train.index],
            index=train.index
        )

        forecast_series = pd.Series(
            [monthly_means[d.month] for d in test.index],
            index=test.index
        )

        metrics = compute_metrics(test['streamflow_cfs'], forecast_series)

        print("\nValidation metrics:")
        for key, value in metrics.items():
            print(f" {key}: {value:.3f}")

        plot_validation(
            train['streamflow_cfs'],
            test['streamflow_cfs'],
            forecast_series,
            metrics,
            model_label,
            train_forecast_cfs=train_fitted,
            save_path=f"{args.model}_validation_plot.png"
        )
        
elif args.model == 'weekly_regression':

    print("\n--- Step 2: Fit/load weekly regression model ---")

    model_file = get_model_file(args.model)

    if REFIT_MODEL or not model_file.exists():
        weekly_model = fit_weekly_regression_model(train, ar_order=args.ar_order)

        print(
            f" Weekly regression model fit using "
            f"{weekly_model['training_rows']:,} training rows."
        )
        print(f" Lag window: {weekly_model['ar_order']} days")

        save_model(weekly_model, args.model)

    else:
        weekly_model = load_model(args.model)

    if not isinstance(weekly_model, dict) or weekly_model.get('model_type') != 'weekly_regression':
        raise TypeError(
            f"{model_file} does not contain a weekly_regression model. "
            "Re-run with --refit True --model weekly_regression to train one."
        )

    model_label = f"Weekly Regression ({weekly_model['ar_order']}-day lag)"

    if RUN_VALIDATION:

        print("\n--- Step 3: Validate weekly regression model ---")

        # Combine train and test so the first test-day prediction can use
        # lagged streamflow values from the end of the training period.
        combined = pd.concat([train, test]).sort_index()

        all_predictions = make_weekly_regression_predictions(weekly_model, combined)

        train_fitted = all_predictions.loc[train.index]
        forecast_series = all_predictions.loc[test.index]

        valid = forecast_series.notna() & test['streamflow_cfs'].notna()

        if valid.sum() == 0:
            raise ValueError("No valid weekly regression predictions were available for validation.")

        metrics = compute_metrics(
            test.loc[valid, 'streamflow_cfs'],
            forecast_series.loc[valid]
        )

        print("\nValidation metrics:")
        for key, value in metrics.items():
            print(f" {key}: {value:.3f}")

        print(
            "\nNSE guide: >0.75 very good | 0.65–0.75 good | "
            "0.50–0.65 satisfactory | <0.50 poor"
        )

        plot_validation(
            train['streamflow_cfs'],
            test['streamflow_cfs'],
            forecast_series,
            metrics,
            model_label,
            train_forecast_cfs=train_fitted,
            save_path=f"{args.model}_validation_plot.png"
        )

print("\nTraining complete.")
