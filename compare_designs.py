"""
compare_designs.py
--------------------
Phase 10 of the AI Aerodynamic Optimizer project.

Phase 9's optimizer only scored shapes at ONE angle of attack (EVAL_ALPHA).
A shape that wins there could still be worse everywhere else -- this script
checks that, by sweeping both the baseline and the optimized airfoil across
a full range of angles and comparing CL, CD, and L/D at every one, not just
the design point. This is the "original vs optimized" comparison your plan
calls for.

Install what you need first (Command Prompt / terminal, not Notepad++):
    pip install neuralfoil numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neuralfoil as nf

# ----------------------- settings you can change -----------------------
BASELINE_DAT = "airfoils/naca2412.dat"
OPTIMIZED_DAT = "optimized_airfoil.dat"     # output of optimize_airfoil.py

ALPHAS = np.arange(-5, 16, 0.5)   # sweep range, in degrees
RE = 1e6
MODEL_SIZE = "large"
DESIGN_ALPHA = 5.0   # whatever EVAL_ALPHA was set to in optimize_airfoil.py
# --------------------------------------------------------------------


def sweep(dat_path, alphas, Re):
    aero = nf.get_aero_from_dat_file(filename=dat_path, alpha=alphas, Re=Re, model_size=MODEL_SIZE)
    cl = np.asarray(aero["CL"])
    cd = np.asarray(aero["CD"])
    confidence = np.asarray(aero["analysis_confidence"])
    l_over_d = np.where(cd > 1e-8, cl / cd, np.nan)
    return pd.DataFrame({
        "alpha": alphas, "CL": cl, "CD": cd,
        "L_over_D": l_over_d, "confidence": confidence,
    })


def row_nearest(df, alpha):
    idx = (df["alpha"] - alpha).abs().idxmin()
    return df.loc[idx]


def main():
    base_df = sweep(BASELINE_DAT, ALPHAS, RE)
    opt_df = sweep(OPTIMIZED_DAT, ALPHAS, RE)
    base_df.to_csv("comparison_baseline.csv", index=False)
    opt_df.to_csv("comparison_optimized.csv", index=False)

    b = row_nearest(base_df, DESIGN_ALPHA)
    o = row_nearest(opt_df, DESIGN_ALPHA)

    print(f"At the design point (alpha={DESIGN_ALPHA}°, Re={RE:.0e}):")
    print(f"{'':12s}{'CL':>8s}{'CD':>10s}{'L/D':>10s}{'confidence':>12s}")
    print(f"{'Baseline':12s}{b['CL']:8.3f}{b['CD']:10.5f}{b['L_over_D']:10.1f}{b['confidence']:12.3f}")
    print(f"{'Optimized':12s}{o['CL']:8.3f}{o['CD']:10.5f}{o['L_over_D']:10.1f}{o['confidence']:12.3f}")
    print(f"\nL/D change at design point: {100 * (o['L_over_D'] / b['L_over_D'] - 1):+.1f}%")

    # how does the optimized shape do across the WHOLE sweep, not just the
    # single point it was optimized for?
    better_count = (opt_df["L_over_D"] > base_df["L_over_D"]).sum()
    print(f"Optimized airfoil beats baseline L/D at {better_count}/{len(ALPHAS)} "
          f"angles across the sweep (not just the design point).")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, ylabel in zip(axes, ["CL", "CD", "L_over_D"], ["CL", "CD", "L/D"]):
        ax.plot(base_df["alpha"], base_df[col], label="baseline")
        ax.plot(opt_df["alpha"], opt_df[col], label="optimized")
        ax.axvline(DESIGN_ALPHA, color="gray", linestyle=":", linewidth=1,
                    label="design point" if col == "CL" else None)
        ax.set_xlabel("Angle of attack (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs angle of attack")
        ax.legend()
    plt.tight_layout()
    plt.savefig("comparison_sweep.png", dpi=150)

    print("\nSaved comparison_baseline.csv, comparison_optimized.csv, comparison_sweep.png")


if __name__ == "__main__":
    main()
