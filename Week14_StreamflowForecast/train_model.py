"""
train_model.py
--------------
Fits the chosen model on the training period and optionally validates on the
test period. Run via run_workflow.sh, or directly:

    python train_model.py --email YOU@EMAIL.COM --pin 1234 [--other-options]
"""

import os
import argparse
import pandas as pd
import hf_hydrodata
from forecast_functions import (
    get_training_test_data,
    fit_longterm_avg_model,
    make_5day_forecast_longterm,
    fit_monthly_avg_model,
    make_5day_forecast_monthly,
    compute_metrics,
    plot_validation,
    save_model,
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
parser.add_argument('--model', default='longterm_avg', choices=['longterm_avg', 'monthly_avg'])
parser.add_argument('--refit',       default='True')
parser.add_argument('--validate',    default='True')
args = parser.parse_args()

REFIT_MODEL    = args.refit.lower()    == 'true'
RUN_VALIDATION = args.validate.lower() == 'true'

hf_hydrodata.register_api_pin(email=args.email, pin=args.pin)

print("\n--- Step 1: Download streamflow data ---")
train, test = get_training_test_data(
    args.gauge_id, args.train_start, args.train_end,
    args.test_start, args.test_end
)

# ── Long-term average model ───────────────────────────────────────────────────
if args.model == 'longterm_avg':
    print("\n--- Step 2: Fit long-term average model ---")
    if REFIT_MODEL or not os.path.exists('saved_model.pkl'):
        mean_flow = fit_longterm_avg_model(train)
        print(f"  Long-term mean: {mean_flow:.2f} cfs")
        save_model(mean_flow)
    else:
        mean_flow = load_model()
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

        print("\n  Generating validation plot ...")
        plot_validation(
            train['streamflow_cfs'], test['streamflow_cfs'],
            forecast_series, metrics, 'Long-term Average',
            train_forecast_cfs=train_fitted
        )
elif args.model == 'monthly_avg':
    print("\n--- Step 2: Fit monthly average model ---")
    monthly_means = fit_monthly_avg_model(train)

    if args.refit == 'True':
        save_model(monthly_means)

    model_label = 'Monthly Average'

    if args.validate == 'True':
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
            print(f"  {key}: {value:.3f}")

        plot_validation(
            train['streamflow_cfs'],
            test['streamflow_cfs'],
            forecast_series,
            metrics,
            model_label,
            train_forecast_cfs=train_fitted
        )
        
print("\nTraining complete.")
