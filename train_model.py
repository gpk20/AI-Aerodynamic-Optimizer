"""
train_model.py
---------------
Phase 5 + Phase 6 of the AI Aerodynamic Optimizer project.

Loads the CSV built by generate_dataset.py and trains a model that predicts
CL and CD from: airfoil shape descriptors + angle of attack + Reynolds
number. Starts with Linear Regression (simplest possible baseline), then
shows how to swap in Random Forest or Gradient Boosting once that works.

Install what you need first (Command Prompt / terminal, not Notepad++):
    pip install pandas scikit-learn matplotlib numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------- settings you can change -----------------------
DATA_CSV = "airfoil_dataset.csv"

FEATURE_COLUMNS = [
    "max_thickness", "max_thickness_loc",
    "max_camber", "max_camber_loc",
    "alpha", "log_reynolds",
]
TARGET_COLUMNS = ["CL", "CD"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Every model here gets trained and compared, same as Phase 5 in your plan.
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}
# --------------------------------------------------------------------


def load_data(path):
    df = pd.read_csv(path)
    df["log_reynolds"] = np.log10(df["reynolds"])

    # A handful of airfoil geometries can produce garbage predictions -- usually
    # a malformed .dat file, or (like multi-element airfoils, e.g. "30p-30n")
    # a shape that isn't a single continuous surface to begin with. Two passes:

    # Pass 1: outright NaN / inf values.
    check_cols = FEATURE_COLUMNS + TARGET_COLUMNS
    df[check_cols] = df[check_cols].replace([np.inf, -np.inf], np.nan)
    bad_rows = df[df[check_cols].isna().any(axis=1)]
    if len(bad_rows):
        print(f"Dropping {len(bad_rows)} row(s) with missing/infinite values "
              f"from: {sorted(bad_rows['airfoil'].unique())}")
        df = df.dropna(subset=check_cols).reset_index(drop=True)

    # Pass 2: finite but physically impossible values (these don't show up as
    # NaN/inf, but can still blow up model training with overflow errors).
    # Generous bounds -- real airfoils never get close to these.
    implausible = (
        (df["max_thickness"] <= 0) | (df["max_thickness"] > 0.6) |
        (df["max_camber"].abs() > 0.3) |
        (df["CL"].abs() > 3) |
        (df["CD"] <= 0) | (df["CD"] > 1)
    )
    if implausible.any():
        print(f"Dropping {implausible.sum()} row(s) with implausible values "
              f"from: {sorted(df.loc[implausible, 'airfoil'].unique())}")
        df = df.loc[~implausible].reset_index(drop=True)

    return df


def train_and_evaluate(df, target, model_name, model):
    X = df[FEATURE_COLUMNS]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    # relative error as a %, same idea as the example table in your project plan
    rel_error_pct = 100 * mae / np.mean(np.abs(y_test))

    return {
        "model": model, "y_test": y_test, "y_pred": y_pred,
        "mae": mae, "rmse": rmse, "r2": r2, "rel_error_pct": rel_error_pct,
    }


def print_comparison_table(all_results):
    print("\n" + "=" * 60)
    print(f"{'Model':<20}{'CL error':>12}{'CD error':>12}{'CL R^2':>10}{'CD R^2':>10}")
    print("-" * 60)
    for model_name in MODELS:
        cl = all_results[(model_name, "CL")]
        cd = all_results[(model_name, "CD")]
        print(f"{model_name:<20}{cl['rel_error_pct']:>11.1f}%{cd['rel_error_pct']:>11.1f}%"
              f"{cl['r2']:>10.3f}{cd['r2']:>10.3f}")
    print("=" * 60)


def plot_actual_vs_predicted(all_results):
    n_models = len(MODELS)
    fig, axes = plt.subplots(n_models, 2, figsize=(11, 5 * n_models))

    for row, model_name in enumerate(MODELS):
        for col, target in enumerate(TARGET_COLUMNS):
            r = all_results[(model_name, target)]
            ax = axes[row, col]
            ax.scatter(r["y_test"], r["y_pred"], alpha=0.4, s=15)
            lims = [min(r["y_test"].min(), r["y_pred"].min()),
                    max(r["y_test"].max(), r["y_pred"].max())]
            ax.plot(lims, lims, "r--", linewidth=1)
            ax.set_xlabel(f"Actual {target}")
            ax.set_ylabel(f"Predicted {target}")
            ax.set_title(f"{model_name} — {target}")

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    print("\nSaved plot to actual_vs_predicted.png")


def main():
    df = load_data(DATA_CSV)
    print(f"Loaded {len(df)} rows, {df['airfoil'].nunique()} airfoils.")

    all_results = {}
    for model_name, model in MODELS.items():
        for target in TARGET_COLUMNS:
            all_results[(model_name, target)] = train_and_evaluate(df, target, model_name, model)

    print_comparison_table(all_results)
    plot_actual_vs_predicted(all_results)


if __name__ == "__main__":
    main()
