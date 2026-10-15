"""
validate_optimized.py
-----------------------
Phase 11 of the AI Aerodynamic Optimizer project.

The optimizer (Phase 8/9) only ever asked ONE source of truth whether a
shape was good: NeuralFoil, evaluated directly on that exact shape. If
NeuralFoil has a blind spot, the optimizer could have driven straight into
it and we'd have no way to tell from inside that same loop.

So this script checks the optimized airfoil's performance a SECOND,
independent way: it retrains a model on your full airfoil_dataset.csv
(the same data train_model.py used, but this model has never seen the
optimized shape specifically) and compares what it predicts against what
NeuralFoil says directly. Think of it like your plan's Phase 11 example --
"AI prediction: 47.6, Simulation: 45.9" -- except here the second opinion
is an independently-trained model instead of a separate physics simulation.
(If you want to go one step further later, the real gold-standard check is
running the optimized shape through actual XFoil -- this script won't
replace that, but it's a genuinely useful check you can run right now with
what you've already built.)

Install what you need first (Command Prompt / terminal, not Notepad++):
    pip install aerosandbox neuralfoil pandas numpy matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aerosandbox as asb
import neuralfoil as nf
from sklearn.ensemble import HistGradientBoostingRegressor

# ----------------------- settings you can change -----------------------
DATASET_CSV = "airfoil_dataset.csv"        # from generate_dataset.py
OPTIMIZED_DAT = "optimized_airfoil.dat"     # from optimize_airfoil.py

ALPHAS = np.arange(-5, 16, 0.5)
RE = 1e6
MODEL_SIZE = "large"

FEATURE_COLUMNS = [
    "max_thickness", "max_thickness_loc",
    "max_camber", "max_camber_loc",
    "alpha", "log_reynolds",
]
RANDOM_STATE = 42
# --------------------------------------------------------------------


def get_shape_descriptors(foil):
    """Same descriptors generate_dataset.py computed for every training airfoil."""
    xs = np.linspace(0.001, 0.999, 300)
    thickness = foil.local_thickness(xs)
    camber = foil.local_camber(xs)
    return {
        "max_thickness": float(thickness.max()),
        "max_thickness_loc": float(xs[np.argmax(thickness)]),
        "max_camber": float(camber.max()),
        "max_camber_loc": float(xs[np.argmax(camber)]),
    }


def train_independent_model(target):
    df = pd.read_csv(DATASET_CSV)
    df["log_reynolds"] = np.log10(df["reynolds"])
    check_cols = FEATURE_COLUMNS + [target]
    df[check_cols] = df[check_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=check_cols)
    # same physical sanity bounds as train_model.py, so this model isn't
    # trained on any garbage rows either
    df = df[(df["max_thickness"] > 0) & (df["max_thickness"] <= 0.6) &
            (df["max_camber"].abs() <= 0.3) & (df["CL"].abs() <= 3) &
            (df["CD"] > 0) & (df["CD"] <= 1)]
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(df[FEATURE_COLUMNS], df[target])
    return model


def main():
    foil = asb.Airfoil(name="optimized", coordinates=OPTIMIZED_DAT)
    shape = get_shape_descriptors(foil)
    print(f"Optimized shape: max_thickness={shape['max_thickness']:.3f}  "
          f"max_camber={shape['max_camber']:.3f}")

    # --- opinion 1: NeuralFoil evaluated directly, same as the optimizer used ---
    direct = nf.get_aero_from_dat_file(filename=OPTIMIZED_DAT, alpha=ALPHAS, Re=RE, model_size=MODEL_SIZE)
    direct_cl, direct_cd = np.asarray(direct["CL"]), np.asarray(direct["CD"])
    direct_conf = np.asarray(direct["analysis_confidence"])

    # --- opinion 2: a model trained on your broad ~1,600-airfoil dataset,
    # that has never specifically seen this shape before ---
    print("Training the independent check model (same data as Phase 5)...")
    cl_model = train_independent_model("CL")
    cd_model = train_independent_model("CD")

    X_query = pd.DataFrame({
        "max_thickness": shape["max_thickness"], "max_thickness_loc": shape["max_thickness_loc"],
        "max_camber": shape["max_camber"], "max_camber_loc": shape["max_camber_loc"],
        "alpha": ALPHAS, "log_reynolds": np.log10(RE),
    })
    model_cl = cl_model.predict(X_query)
    model_cd = cd_model.predict(X_query)

    df = pd.DataFrame({
        "alpha": ALPHAS,
        "CL_neuralfoil": direct_cl, "CL_independent_model": model_cl,
        "CD_neuralfoil": direct_cd, "CD_independent_model": model_cd,
        "neuralfoil_confidence": direct_conf,
    })

    # percentage differences blow up (even go to absurd numbers) wherever the
    # NeuralFoil denominator is tiny/non-physical -- a real risk at extreme,
    # low-confidence angles. Only compute a percentage where CD is a sane,
    # physically meaningful value; report the rest as a plain (unscaled)
    # difference instead, which can't blow up this way.
    cd_sane = df["CD_neuralfoil"].between(1e-3, 1)
    df["CL_diff_pct"] = np.where(
        df["CL_neuralfoil"].abs() > 1e-2,
        100 * (df["CL_independent_model"] - df["CL_neuralfoil"]) / df["CL_neuralfoil"], np.nan)
    df["CD_diff_pct"] = np.where(
        cd_sane, 100 * (df["CD_independent_model"] - df["CD_neuralfoil"]) / df["CD_neuralfoil"], np.nan)
    df["CD_diff_abs"] = (df["CD_independent_model"] - df["CD_neuralfoil"]).abs()
    df.to_csv("validation_comparison.csv", index=False)

    n_excluded = (~cd_sane).sum()
    mean_cl_diff = df["CL_diff_pct"].abs().mean()
    mean_cd_diff = df["CD_diff_pct"].abs().mean()
    mean_cd_abs = df["CD_diff_abs"].mean()
    mean_conf = df["neuralfoil_confidence"].mean()

    print(f"\nAverage disagreement across the sweep:")
    print(f"  CL: {mean_cl_diff:.1f}%    CD: {mean_cd_diff:.1f}% (abs: {mean_cd_abs:.5f})")
    if n_excluded:
        print(f"  ({n_excluded} angle(s) excluded from CD % -- NeuralFoil's own CD there "
              f"was non-physical, usually a sign of a low-confidence, out-of-range angle)")
    print(f"Average NeuralFoil confidence on this shape: {mean_conf:.3f}")

    if mean_cl_diff < 10 and mean_cd_diff < 20:
        print("\n-> The two independent predictions broadly agree. That's a decent "
              "sign the optimized shape's performance is real, not an artifact of "
              "one model's blind spot -- though it's still not the same as a real "
              "XFoil run or wind-tunnel data.")
    else:
        print("\n-> The two predictions diverge notably. That's a sign the optimized "
              "shape may sit outside what your training data covers well -- worth "
              "treating the optimizer's L/D claim with real skepticism, and a good "
              "thing to discuss honestly in your Phase 13 write-up.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col_a, col_b, ylabel in [
        (axes[0], "CL_neuralfoil", "CL_independent_model", "CL"),
        (axes[1], "CD_neuralfoil", "CD_independent_model", "CD"),
    ]:
        ax.plot(df["alpha"], df[col_a], label="NeuralFoil (direct)")
        ax.plot(df["alpha"], df[col_b], label="independent model", linestyle="--")
        ax.set_xlabel("Angle of attack (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}: two independent predictions")
        ax.legend()
    plt.tight_layout()
    plt.savefig("validation_comparison.png", dpi=150)
    print("\nSaved validation_comparison.csv and validation_comparison.png")


if __name__ == "__main__":
    main()
