"""
CS 4375 - Assignment 1, Part 1
Linear regression with gradient descent.

Dataset: UCI Air Quality (AirQualityUCI.csv)
Target = C6H6(GT)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/SGcoder120/ML-assignment-1/main/AirQualityUCI.csv"
TARGET = "C6H6(GT)"

# PT08.S2(NMHC) basically measures the same thing as the target (corr ~0.98), so it gets dropped
LEAKY_FEATURE = "PT08.S2(NMHC)"

MISSING_SENTINEL = -200  # UCI uses -200 for missing readings, not NaN

# This loads and cleans the Air Quality dataset, yielding a scaled train/test split ready for training.
class DataPreprocessor:
    """Loads and cleans the Air Quality dataset for regression on C6H6(GT)."""

    def __init__(self, url: str = DATA_URL, target: str = TARGET,
                 drop_leaky_feature: bool = True):
        self.url = url
        self.target = target
        self.drop_leaky_feature = drop_leaky_feature
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.feature_names: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None

    def load(self) -> pd.DataFrame:
        # semicolon-delimited, comma decimals (e.g. "2,6")
        self.raw_df = pd.read_csv(self.url, sep=';', decimal=',')
        return self.raw_df

    def clean(self) -> pd.DataFrame:
        assert self.raw_df is not None, "call load() before clean()"
        df = self.raw_df.copy()

        # trailing semicolons in the source file leave two empty columns
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
        df = df.drop(columns=unnamed_cols, errors='ignore')
        df = df.dropna(how='all', subset=['Date'])  # blank trailing rows

        n_before = len(df)

        df = df.drop(columns=['NMHC(GT)'], errors='ignore')  # ~90% missing, unusable
        df = df.replace(MISSING_SENTINEL, np.nan)
        df = df.dropna()
        n_after_na = len(df)

        df = df.drop_duplicates()
        n_after_dupes = len(df)

        print(f"Rows before cleaning: {n_before}")
        print(f"Rows after dropping missing (-200 sentinel) values: {n_after_na}")
        print(f"Rows after dropping duplicates: {n_after_dupes}")

        self.df = df
        return self.df

    def drop_non_features(self) -> pd.DataFrame:
        # Date/Time as raw strings aren't usable in a linear model.
        # Could engineer hour-of-day as a numeric feature instead -- skipped for now.
        assert self.df is not None, "call clean() before drop_non_features()"
        df = self.df.copy()
        df = df.drop(columns=['Date', 'Time'], errors='ignore')

        self.df = df
        return self.df

    def drop_uncorrelated_features(self, threshold: float = 0.05) -> pd.DataFrame:
        assert self.df is not None, "call drop_non_features() before drop_uncorrelated_features()"
        df = self.df.copy()
        corr_with_target = df.corr(numeric_only=True)[self.target].drop(self.target)
        weak_features = corr_with_target[corr_with_target.abs() < threshold].index.tolist()

        print(f"Dropping weakly-correlated features (|corr| < {threshold}): {weak_features}")
        df = df.drop(columns=weak_features)

        self.df = df
        return self.df

    def finalize(self) -> pd.DataFrame:
        assert self.df is not None, "call drop_uncorrelated_features() before finalize()"
        df = self.df.copy()
        if self.drop_leaky_feature:
            df = df.drop(columns=[LEAKY_FEATURE], errors='ignore')

        self.df = df
        self.feature_names = [c for c in df.columns if c != self.target]
        return self.df

    def run(self) -> pd.DataFrame:
        self.load()
        self.clean()
        self.drop_non_features()
        self.drop_uncorrelated_features()
        self.finalize()
        assert self.df is not None
        return self.df

    def split_and_scale(self, test_size: float = 0.2, random_state: int = 42):
        # scaler fit on train only, then reused on test -- avoids leaking test stats
        assert self.df is not None and self.feature_names is not None, \
            "call run() before split_and_scale()"
        X = self.df[self.feature_names].values
        y = self.df[self.target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scaler = scaler
        return X_train_scaled, X_test_scaled, y_train, y_test

# This contains a linear regression model trained with gradient descent.
class LinearRegressionGD:
    """
    y_hat = X.w + b, loss = MSE. Stops when MSE improvement drops below
    `tol`, or after `max_iterations`, whichever comes first.
    """

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 10000,
                 tol: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.mse_history: List[float] = []
        self.n_iterations_run = 0

    def _mse(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0.0
        self.mse_history = []

        prev_mse = None
        for i in range(self.max_iterations):
            y_pred = X.dot(weights) + bias
            error = y_pred - y

            grad_w = (2 / n_samples) * X.T.dot(error)
            grad_b = (2 / n_samples) * np.sum(error)

            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b

            current_mse = self._mse(y, X.dot(weights) + bias)
            self.mse_history.append(current_mse)

            if prev_mse is not None and abs(prev_mse - current_mse) < self.tol:
                self.n_iterations_run = i + 1
                break
            prev_mse = current_mse
        else:
            self.n_iterations_run = self.max_iterations

        if not np.isfinite(current_mse):
            print(f"WARNING: training diverged (lr={self.learning_rate}). "
                  f"Try a smaller learning rate.")

        self.weights = weights
        self.bias = bias
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.weights is not None and self.bias is not None, \
            "call fit() before predict()"
        return X.dot(self.weights) + self.bias


# Computes MSE, RMSE, MAE, R2, Explained Variance, Max Error, and (optionally) Adjusted R2.
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_features: Optional[int] = None) -> dict:
    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    explained_variance = 1 - np.var(residuals) / np.var(y_true)
    max_err = np.max(np.abs(residuals))

    metrics = {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
        "ExplainedVariance": explained_variance, "MaxError": max_err,
    }

    if n_features is not None:
        n = len(y_true)
        metrics["AdjustedR2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return metrics


# Grid-searches learning_rate/max_iterations, logs every trial, and holds onto the best model plus the full trial history for plotting.
class ModelTrainer:
    def __init__(self, learning_rates, max_iterations_list, tol: float = 1e-6,
                 log_path: str = "trials_log.txt"):
        self.learning_rates = learning_rates
        self.max_iterations_list = max_iterations_list
        self.tol = tol
        self.log_path = log_path
        self.trials: List[dict] = []
        self.best_model: Optional[LinearRegressionGD] = None
        self.best_config: Optional[dict] = None

    def tune(self, X_train, y_train, X_test, y_test):
        best_test_mse = np.inf
        self.trials = []

        with open(self.log_path, "w") as log_file:
            log_file.write("learning_rate,max_iterations,tol,iterations_run,train_mse,test_mse\n")

            for lr in self.learning_rates:
                for max_iter in self.max_iterations_list:
                    model = LinearRegressionGD(learning_rate=lr, max_iterations=max_iter, tol=self.tol)
                    model.fit(X_train, y_train)

                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    train_mse = np.mean((train_pred - y_train) ** 2)
                    test_mse = np.mean((test_pred - y_test) ** 2)

                    log_line = f"{lr},{max_iter},{self.tol},{model.n_iterations_run},{train_mse:.6f},{test_mse:.6f}"
                    log_file.write(log_line + "\n")
                    print(log_line)

                    self.trials.append({
                        "learning_rate": lr, "max_iterations": max_iter,
                        "train_mse": train_mse, "test_mse": test_mse,
                    })

                    if np.isfinite(test_mse) and test_mse < best_test_mse:
                        best_test_mse = test_mse
                        self.best_model = model
                        self.best_config = {"learning_rate": lr, "max_iterations": max_iter, "tol": self.tol}

        print(f"\nBest config: {self.best_config} -> test MSE = {best_test_mse:.6f}")
        assert self.best_model is not None and self.best_config is not None, \
            "every trial diverged -- lower the learning rates or widen the grid"
        return self.best_model, self.best_config

    def plot_mse_history(self, path: str = "mse_vs_iterations.png"):
        assert self.best_model is not None
        plt.figure(figsize=(7, 5))
        plt.plot(self.best_model.mse_history)
        plt.xlabel("Iteration")
        plt.ylabel("Training MSE")
        plt.title("MSE vs. Iterations (Best Model)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")

    def plot_actual_vs_predicted(self, y_true, y_pred, path: str = "actual_vs_predicted.png"):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, 'r--', label="Perfect prediction")
        plt.xlabel(f"Actual {TARGET}")
        plt.ylabel(f"Predicted {TARGET}")
        plt.title(f"Actual vs. Predicted {TARGET} (Test Set)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")

    def plot_residuals(self, y_true, y_pred, path: str = "residuals.png"):
        # residual vs predicted -- random scatter around 0 means no systematic bias
        residuals = y_true - y_pred
        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel(f"Predicted {TARGET}")
        plt.ylabel("Residual (actual - predicted)")
        plt.title("Residual Plot (Test Set)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")

    def plot_coefficients(self, feature_names: List[str], path: str = "coefficients.png"):
        assert self.best_model is not None and self.best_model.weights is not None
        weights = self.best_model.weights
        colors = ['tab:blue' if w >= 0 else 'tab:red' for w in weights]

        plt.figure(figsize=(8, 5))
        plt.bar(feature_names, weights, color=colors)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Weight (on standardized features)")
        plt.title("Fitted Regression Weights")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")

    def plot_tuning_results(self, path: str = "tuning_results.png"):
        # test MSE vs learning rate, one line per max_iterations -- shows the actual search
        assert self.trials, "call tune() first"
        max_iter_values = sorted(set(t["max_iterations"] for t in self.trials))

        plt.figure(figsize=(7, 5))
        for max_iter in max_iter_values:
            subset = [t for t in self.trials if t["max_iterations"] == max_iter]
            subset.sort(key=lambda t: t["learning_rate"])
            lrs = [t["learning_rate"] for t in subset]
            test_mses = [t["test_mse"] for t in subset]
            plt.plot(lrs, test_mses, marker='o', label=f"max_iter={max_iter}")

        plt.xscale('log')
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Test MSE")
        plt.title("Hyperparameter Search: Test MSE vs. Learning Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")


def plot_features_vs_target(df: pd.DataFrame, features: list, target: str = TARGET, path: str = "features_vs_target.png"):
    n = len(features)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, feature in zip(axes, features):
        ax.scatter(df[feature], df[target], alpha=0.4, s=10)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f"{target} vs. {feature}")

    for ax in axes[len(features):]:
        ax.axis("off")  # unused grid slots

    fig.suptitle(f"{target} vs. Multiple Input Features", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved plot: {path}")

# Runs the first pipline
if __name__ == "__main__":
    prep = DataPreprocessor()
    df = prep.run()
    print("\nFinal preprocessed shape:", df.shape)
    print("Final feature columns:", prep.feature_names)
    assert prep.feature_names is not None

    X_train, X_test, y_train, y_test = prep.split_and_scale()
    print("\nX_train shape:", X_train.shape, "| X_test shape:", X_test.shape)

    trainer = ModelTrainer(
        learning_rates=[0.001, 0.01, 0.05, 0.1, 0.3],
        max_iterations_list=[1000, 5000, 10000],
        tol=1e-6,
        log_path="trials_log.txt",
    )
    best_model, best_config = trainer.tune(X_train, y_train, X_test, y_test)

    assert best_model.weights is not None and best_model.bias is not None
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    n_features = len(prep.feature_names)
    train_metrics = evaluate(y_train, train_pred, n_features=n_features)
    test_metrics = evaluate(y_test, test_pred, n_features=n_features)

    print("\nBest Model:")
    print("Config:", best_config)
    print("Iterations actually run:", best_model.n_iterations_run)
    print("Weights:", dict(zip(prep.feature_names, np.round(best_model.weights, 4))))
    print("Bias:", round(best_model.bias, 4))
    print("Train metrics:", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    trainer.plot_mse_history()
    trainer.plot_actual_vs_predicted(y_test, test_pred)
    trainer.plot_residuals(y_test, test_pred)
    trainer.plot_coefficients(prep.feature_names)
    trainer.plot_tuning_results()
    plot_features_vs_target(df, prep.feature_names)