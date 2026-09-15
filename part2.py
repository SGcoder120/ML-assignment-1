"""
CS 4375 - Assignment 1, Part 2
Same dataset, preprocessing, and split as Part 1 -- here scikit-learn's
SGDRegressor builds the model instead of the hand-rolled version.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error

from part1 import DataPreprocessor, TARGET

# Computes MSE, RMSE, MAE, R2, Explained Variance, Max Error, and (optionally) Adjusted R2 via sklearn.
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_features: Optional[int] = None) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "R2": r2,
        "ExplainedVariance": explained_variance_score(y_true, y_pred),
        "MaxError": max_error(y_true, y_pred),
    }

    if n_features is not None:
        n = len(y_true)
        metrics["AdjustedR2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return metrics

# Grid-searches eta0/alpha/max_iter, logs every trial, returns the best model
# by test MSE plus the full trial history (used later for the tuning plot).
def tune_hyperparameters(X_train, y_train, X_test, y_test, learning_rates, alphas, max_iter_list, log_path="trials_log_part2.txt"):
    # grid search over eta0 x alpha x max_iter, logs every trial, keeps best test MSE
    best_model: Optional[SGDRegressor] = None
    best_config = None
    best_test_mse = np.inf
    trials = []

    with open(log_path, "w") as log_file:
        log_file.write("eta0,alpha,max_iter,n_iter_actual,train_mse,test_mse\n")

        for eta0 in learning_rates:
            for alpha in alphas:
                for max_iter in max_iter_list:
                    model = SGDRegressor(
                        loss="squared_error",
                        penalty="l2",
                        alpha=alpha,
                        learning_rate="constant",
                        eta0=eta0,
                        max_iter=max_iter,
                        tol=1e-6,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)

                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    train_mse = mean_squared_error(y_train, train_pred)
                    test_mse = mean_squared_error(y_test, test_pred)

                    log_line = (f"{eta0},{alpha},{max_iter},{model.n_iter_},"
                                f"{train_mse:.6f},{test_mse:.6f}")
                    log_file.write(log_line + "\n")
                    print(log_line)

                    trials.append({
                        "eta0": eta0, "alpha": alpha, "max_iter": max_iter,
                        "train_mse": train_mse, "test_mse": test_mse,
                    })

                    if np.isfinite(test_mse) and test_mse < best_test_mse:
                        best_test_mse = test_mse
                        best_model = model
                        best_config = {"eta0": eta0, "alpha": alpha, "max_iter": max_iter}

    print(f"\nBest config: {best_config} -> test MSE = {best_test_mse:.6f}")
    assert best_model is not None and best_config is not None, \
        "every trial diverged -- lower the learning rates or widen the grid"
    return best_model, best_config, trials


def plot_loss_curve(X_train, y_train, best_config, path="sgd_loss_vs_iterations.png"):
    # SGDRegressor has no built-in loss history, so rebuild it with partial_fit
    # to get an MSE-vs-iterations curve comparable to Part 1's
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=best_config["alpha"],
        learning_rate="constant",
        eta0=best_config["eta0"],
        random_state=42,
    )

    mse_history = []
    n_passes = 200
    for _ in range(n_passes):
        model.partial_fit(X_train, y_train)
        pred = model.predict(X_train)
        mse_history.append(mean_squared_error(y_train, pred))

    plt.figure(figsize=(7, 5))
    plt.plot(mse_history)
    plt.xlabel("Pass over training data")
    plt.ylabel("Training MSE")
    plt.title("SGDRegressor: MSE vs. Iterations (Best Config)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")
    return mse_history


def plot_actual_vs_predicted(y_true, y_pred, path="sgd_actual_vs_predicted.png"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', label="Perfect prediction")
    plt.xlabel(f"Actual {TARGET}")
    plt.ylabel(f"Predicted {TARGET}")
    plt.title(f"SGDRegressor: Actual vs. Predicted {TARGET} (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_residuals(y_true, y_pred, path="sgd_residuals.png"):
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel(f"Predicted {TARGET}")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("SGDRegressor: Residual Plot (Test Set)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_coefficients(model: SGDRegressor, feature_names, path="sgd_coefficients.png"):
    weights = model.coef_
    colors = ['tab:blue' if w >= 0 else 'tab:red' for w in weights]

    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, weights, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Weight (on standardized features)")
    plt.title("SGDRegressor: Fitted Regression Weights")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")


def plot_tuning_results(trials, path="sgd_tuning_results.png"):
    # Test MSE vs eta0, one line per max_iter, at the best alpha found
    best_alpha = min(trials, key=lambda t: t["test_mse"])["alpha"]
    subset = [t for t in trials if t["alpha"] == best_alpha]
    max_iter_values = sorted(set(t["max_iter"] for t in subset))

    plt.figure(figsize=(7, 5))
    for max_iter in max_iter_values:
        group = [t for t in subset if t["max_iter"] == max_iter]
        group.sort(key=lambda t: t["eta0"])
        etas = [t["eta0"] for t in group]
        test_mses = [t["test_mse"] for t in group]
        plt.plot(etas, test_mses, marker='o', label=f"max_iter={max_iter}")

    plt.xscale('log')
    plt.xlabel("eta0 / Learning Rate (log scale)")
    plt.ylabel("Test MSE")
    plt.title(f"SGDRegressor Tuning: Test MSE vs. eta0 (alpha={best_alpha})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved plot: {path}")

# Runs the second pipeline
if __name__ == "__main__":
    prep = DataPreprocessor()
    df = prep.run()
    print("\nFinal preprocessed shape:", df.shape)
    print("Final feature columns:", prep.feature_names)
    assert prep.feature_names is not None

    X_train, X_test, y_train, y_test = prep.split_and_scale()
    print("\nX_train shape:", X_train.shape, "| X_test shape:", X_test.shape)

    learning_rates = [0.001, 0.01, 0.05, 0.1]
    alphas = [0.0001, 0.001, 0.01]
    max_iter_list = [1000, 5000]

    best_model, best_config, trials = tune_hyperparameters(
        X_train, y_train, X_test, y_test,
        learning_rates=learning_rates,
        alphas=alphas,
        max_iter_list=max_iter_list,
        log_path="trials_log_part2.txt",
    )

    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    n_features = len(prep.feature_names)
    train_metrics = evaluate(y_train, train_pred, n_features=n_features)
    test_metrics = evaluate(y_test, test_pred, n_features=n_features)

    print("\nBest SGDRegressor Model:")
    print("Config:", best_config)
    print("n_iter_ (actual iterations run):", best_model.n_iter_)
    print("Weight coefficients:", dict(zip(prep.feature_names, np.round(best_model.coef_, 4))))
    print("Intercept:", round(best_model.intercept_[0], 4))
    print("Train metrics:", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    plot_loss_curve(X_train, y_train, best_config)
    plot_actual_vs_predicted(y_test, test_pred)
    plot_residuals(y_test, test_pred)
    plot_coefficients(best_model, prep.feature_names)
    plot_tuning_results(trials)

    # Compare test MSE across seeds -- close agreement = stable solution, not a lucky draw
    print("\nStability check across random seeds:")
    for seed in [0, 1, 2]:
        check_model = SGDRegressor(
            loss="squared_error", penalty="l2",
            alpha=best_config["alpha"], learning_rate="constant",
            eta0=best_config["eta0"], max_iter=best_config["max_iter"],
            tol=1e-6, random_state=seed,
        )
        check_model.fit(X_train, y_train)
        check_test_mse = mean_squared_error(y_test, check_model.predict(X_test))
        print(f"random_state={seed} -> test MSE = {check_test_mse:.4f}")