"""
generate_dataset.py
--------------------
Phase 3 + Phase 4 of the AI Aerodynamic Optimizer project.

Your UIUC .dat files only contain GEOMETRY (the x,y coordinates that trace
the airfoil's outline). They do NOT contain CL, CD, angle of attack, or
Reynolds number -- that performance data has to be generated separately,
normally by running each airfoil through an aerodynamics solver like XFoil.

This script does that for you automatically using NeuralFoil, a neural
network that was trained on tens of millions of real XFoil runs and gives
near-identical results in a fraction of the time (no crashes, no manual
XFoil installation needed). For every airfoil in your folder, it:

  1. Loads the airfoil geometry from its .dat file
  2. Computes a few simple numeric shape descriptors (Phase 4: turning the
     shape into numbers an ML model can use)
  3. Runs it through NeuralFoil across a grid of angles of attack and
     Reynolds numbers (Phase 3: generating the performance data)
  4. Saves everything to one big CSV -- this CSV is what train_model.py
     (Phase 5) will learn from.

Install what you need first (Command Prompt / terminal, not Notepad++):
    pip install neuralfoil aerosandbox numpy pandas
"""

import numpy as np
import pandas as pd
import aerosandbox as asb
import neuralfoil as nf
from pathlib import Path

# ----------------------- settings you can change -----------------------
DAT_FOLDER = Path("airfoils")          # folder containing your UIUC .dat files
OUTPUT_CSV = Path("airfoil_dataset.csv")

ALPHAS = np.arange(-4, 16, 1)          # angles of attack to test, in degrees
REYNOLDS = [2e5, 5e5, 1e6, 2e6]        # Reynolds numbers to test
MODEL_SIZE = "large"                   # NeuralFoil model: xxsmall..xxxlarge
# --------------------------------------------------------------------


def get_shape_descriptors(foil: asb.Airfoil) -> dict:
    """Turn an airfoil's geometry into a handful of numbers (Phase 4)."""
    xs = np.linspace(0.001, 0.999, 300)
    thickness = foil.local_thickness(xs)
    camber = foil.local_camber(xs)
    return {
        "max_thickness": float(thickness.max()),
        "max_thickness_loc": float(xs[np.argmax(thickness)]),
        "max_camber": float(camber.max()),
        "max_camber_loc": float(xs[np.argmax(camber)]),
    }


def main():
    dat_files = sorted(DAT_FOLDER.glob("*.dat"))
    if not dat_files:
        raise SystemExit(f"No .dat files found in {DAT_FOLDER.resolve()}")

    print(f"Found {len(dat_files)} airfoil files.")
    rows = []
    failed = []

    for i, path in enumerate(dat_files, 1):
        name = path.stem
        print(f"[{i}/{len(dat_files)}] {name}")
        try:
            foil = asb.Airfoil(name=name, coordinates=str(path))
            shape = get_shape_descriptors(foil)
        except Exception as e:
            print(f"    could not read geometry, skipping: {e}")
            failed.append((name, str(e)))
            continue

        for Re in REYNOLDS:
            try:
                aero = nf.get_aero_from_dat_file(
                    filename=str(path),
                    alpha=ALPHAS,
                    Re=Re,
                    model_size=MODEL_SIZE,
                )
            except Exception as e:
                print(f"    NeuralFoil failed at Re={Re:.0e}, skipping: {e}")
                failed.append((name, f"Re={Re}: {e}"))
                continue

            for j, alpha in enumerate(ALPHAS):
                cl = float(aero["CL"][j])
                cd = float(aero["CD"][j])
                rows.append({
                    "airfoil": name,
                    **shape,
                    "alpha": float(alpha),
                    "reynolds": Re,
                    "CL": cl,
                    "CD": cd,
                    "L_over_D": cl / cd if cd > 1e-8 else np.nan,
                    "analysis_confidence": float(aero["analysis_confidence"][j]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} rows from {df['airfoil'].nunique()} airfoils to {OUTPUT_CSV.resolve()}")
    if failed:
        print(f"{len(failed)} airfoil/Re combinations failed and were skipped:")
        for name, err in failed[:10]:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
