# Homework 7: Streamflow Forecasting Workflow

This repository contains my Homework 7 streamflow forecasting workflow for the Verde River near Camp Verde, Arizona using USGS gauge `09506000`.

The workflow downloads daily streamflow data, optionally trains and validates a model, and generates a 5-day daily average streamflow forecast starting on a user-selected date.

## Repository contents

```text
Homework_7/
├── README.md
├── activities.md
├── environment.yml
└── Week14_StreamflowForecast/
    ├── run_workflow.sh
    ├── train_model.py
    ├── generate_forecast.py
    ├── forecast_functions.py
    ├── models/
    │   ├── longterm_avg_model.pkl
    │   ├── monthly_avg_model.pkl
    │   └── weekly_regression_model.pkl
    └── figures/
        ├── longterm_avg_validation_plot.png
        ├── longterm_avg_forecast_plot.png
        ├── monthly_avg_validation_plot.png
        ├── monthly_avg_forecast_plot.png
        ├── weekly_regression_validation_plot.png
        └── weekly_regression_forecast_plot.png
```

## Environment setup

This workflow uses Python, NumPy, pandas, matplotlib, Jupyter tools, and `hf_hydrodata`.

To create the conda environment from the included `environment.yml` file, run this from the repository root:

```bash
conda env create -f environment.yml
conda activate hw7_forecast
```

If the environment already exists and you want to rebuild it from scratch:

```bash
conda deactivate
conda env remove -n hw7_forecast
conda env create -f environment.yml
conda activate hw7_forecast
```

## Running the workflow

From the repository root, move into the workflow folder:

```bash
cd Week14_StreamflowForecast
```

Make sure the shell script is executable:

```bash
chmod +x run_workflow.sh
```

Then run the workflow:

```bash
./run_workflow.sh
```

The script will ask for a HydroFrame email and PIN:

```text
HydroFrame email:
HydroFrame PIN:
```

The PIN is hidden when typed into the terminal.

## Main user options

The main workflow settings are located near the top of `run_workflow.sh`.

```bash
GAUGE_ID="09506000"
AR_ORDER=7
TRAIN_START="1990-01-01"
TRAIN_END="2022-12-31"
TEST_START="2023-01-01"
TEST_END="2024-12-31"
FORECAST_DATE="2024-04-30"
REFIT_MODEL="True"
RUN_VALIDATION="True"
MODEL="weekly_regression"
```

| Option | Meaning |
|---|---|
| `GAUGE_ID` | USGS gauge ID. The default is `09506000`, Verde River near Camp Verde, AZ. |
| `AR_ORDER` | Number of previous days used by the weekly regression model. The default is 7. |
| `TRAIN_START` / `TRAIN_END` | Date range used to train the selected model. |
| `TEST_START` / `TEST_END` | Date range used to validate the selected model. |
| `FORECAST_DATE` | First day of the 5-day forecast. |
| `REFIT_MODEL` | `True` fits the model again. `False` loads the saved model from the `models/` folder. |
| `RUN_VALIDATION` | `True` prints validation metrics and saves a validation plot. |
| `MODEL` | Model choice. Options are `longterm_avg`, `monthly_avg`, and `weekly_regression`. |

## Model choices

This workflow includes three model options.

### 1. `longterm_avg`

The long-term average model predicts the same streamflow value for every forecast day.

It calculates the mean streamflow over the full training period and uses that one value for the entire forecast. This is the simplest baseline model. It does not account for seasonality or recent streamflow conditions.

### 2. `monthly_avg`

The monthly average model calculates a separate historical mean streamflow for each calendar month.

For example, a forecast day in April receives the historical April mean, and a forecast day in May receives the historical May mean. This model improves on the long-term average model because it includes seasonal differences in streamflow.

### 3. `weekly_regression`

The weekly regression model is the additional model I added for this homework assignment.

This model predicts streamflow using the previous 7 days of streamflow. It uses lagged log-transformed streamflow values:

```text
log(Q_t + 1) = b0
             + b1 log(Q_{t-1} + 1)
             + b2 log(Q_{t-2} + 1)
             + ...
             + b7 log(Q_{t-7} + 1)
```

where:

- `Q_t` is the streamflow being predicted for the current day.
- `Q_{t-1}` through `Q_{t-7}` are the streamflows from the previous 1 through 7 days.
- `b0` is the fitted intercept.
- `b1` through `b7` are the fitted regression coefficients.

For the 5-day forecast, the model works recursively. Day 1 uses the most recent observed streamflow values. Day 2 uses the Day 1 forecast as part of its lag history, and the same pattern continues through Day 5.

This model is different from the two average models because it uses recent flow conditions rather than only historical averages.

## Outputs

The workflow creates two main types of outputs: saved model files and figures.

### Saved models

Model files are saved in the `models/` folder:

```text
models/longterm_avg_model.pkl
models/monthly_avg_model.pkl
models/weekly_regression_model.pkl
```

Using model-specific files avoids overwriting one model with another model.

### Figures

Plots are saved in the `figures/` folder:

```text
figures/longterm_avg_validation_plot.png
figures/longterm_avg_forecast_plot.png
figures/monthly_avg_validation_plot.png
figures/monthly_avg_forecast_plot.png
figures/weekly_regression_validation_plot.png
figures/weekly_regression_forecast_plot.png
```

The validation plots compare observed and predicted streamflow for the test period. The forecast plots show recent observed streamflow and the 5-day forecast.

## Validation metrics

When `RUN_VALIDATION="True"`, the workflow prints model-fit metrics.

| Metric | Meaning |
|---|---|
| RMSE | Root mean squared error in cubic feet per second. Lower values are better. |
| R2 | Squared correlation between observed and predicted values. Higher values are better. |
| NSE | Nash-Sutcliffe efficiency. Values closer to 1 are better. |

The script also prints this general NSE guide:

```text
>0.75 very good | 0.65–0.75 good | 0.50–0.65 satisfactory | <0.50 poor
```

The long-term average model may produce an undefined `R2` value because it predicts a constant value. This does not mean the workflow failed. It happens because a constant prediction series has no variance for the correlation calculation.

## Workflow summary

The workflow is controlled by `run_workflow.sh`.

1. The user edits the options in `run_workflow.sh`.
2. The script asks for HydroFrame login information.
3. If `REFIT_MODEL="True"` or `RUN_VALIDATION="True"`, the script runs `train_model.py`.
4. `train_model.py` downloads streamflow data, fits or loads the selected model, and optionally validates it.
5. The script then runs `generate_forecast.py`.
6. `generate_forecast.py` downloads recent streamflow data, loads the selected saved model, and creates a 5-day forecast.
7. Model files are saved to `models/`.
8. Plot files are saved to `figures/`.

## Changes and improvements I made

In addition to completing the class activity models, I made the following updates:

1. Added a third model called `weekly_regression`.
2. Added model-specific save files so each model has its own `.pkl` file.
3. Added a `models/` folder for saved model outputs.
4. Added a `figures/` folder for validation and forecast plots.
5. Updated the workflow so plots are saved with model-specific filenames.
6. Added forecast-date checking so the workflow warns the user if the selected forecast date is later than the latest available streamflow data.
7. Tested the workflow with both refitting and loading existing saved models.

## AI-use note

I used AI to help plan and debug this workflow. AI helped me think through how to organize the model-saving folder, figure outputs, and the added weekly regression model. I reviewed and tested the code by running the workflow for the long-term average, monthly average, and weekly regression models.

## Notes

The main workflow is in:

```text
Week14_StreamflowForecast/run_workflow.sh
```

The main functions are in:

```text
Week14_StreamflowForecast/forecast_functions.py
```

The model training and validation script is:

```text
Week14_StreamflowForecast/train_model.py
```

The forecast-generation script is:

```text
Week14_StreamflowForecast/generate_forecast.py
```
