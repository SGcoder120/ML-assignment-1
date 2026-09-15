# CS 4375 - Assignment 1: Linear Regression using Gradient Descent

## Dataset

This project uses the UCI Air Quality dataset, hosted on GitHub:
https://raw.githubusercontent.com/SGcoder120/ML-assignment-1/main/AirQualityUCI.csv

We're predicting `C6H6(GT)`, the benzene level in the air.

## Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn

Install everything with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Files

- `part1.py` - cleans the data and builds a linear regression model by
  hand, using gradient descent (no built-in regression functions). Tries
  out different settings to find the best one, keeps a log of every
  attempt, and creates several charts.
- `part2.py` - uses the same data as Part 1 (imported from `part1.py`),
  but builds the model using scikit-learn's `SGDRegressor` instead of
  writing the math by hand. Also tries different settings, logs the
  results, and creates charts.

## How to run

Keep both files in the same folder, since `part2.py` needs `part1.py` to run.

```bash
python3 part1.py
python3 part2.py
```

Each script downloads the data, cleans it, tests different settings to
find the best model, prints the results, and saves a log file plus some
charts in the same folder.

## Files each script creates

Running `part1.py` creates:

- `trials_log.txt`
- `mse_vs_iterations.png`
- `actual_vs_predicted.png`
- `residuals.png`
- `coefficients.png`
- `tuning_results.png`
- `features_vs_target.png`

Running `part2.py` creates:

- `trials_log_part2.txt`
- `sgd_loss_vs_iterations.png`
- `sgd_actual_vs_predicted.png`
- `sgd_residuals.png`
- `sgd_coefficients.png`
- `sgd_tuning_results.png`

## What Part 2 uses from scikit-learn

- `SGDRegressor` - the library's built-in model, standing in for the
  by-hand version from Part 1
- A few functions from `sklearn.metrics` to score how good the
  predictions are
- `numpy` and `matplotlib`, same as in Part 1

## References

- Claude (Anthropic), Sonnet 4.5, September 2026. Used for: discussing
  which dataset features to use, fixing some code errors, explaining
  the gradient descent math, and help organizing the code. All decisions
  and the final code were reviewed and understood by me.
