"""
generate_airfoils.py
---------------------
Phase 7 of the AI Aerodynamic Optimizer project.

Creates NEW airfoil shapes rather than just analyzing existing ones. It does
this by representing an airfoil as a short list of ~18 smooth numbers (a
"Kulfan"/CST parameterization: roughly, 8 numbers shaping the upper surface,
8 shaping the lower surface, plus a leading-edge and trailing-edge number)
instead of hundreds of raw x,y coordinates. Nudging those numbers produces
smooth, valid new shapes -- no risk of a jagged or self-intersecting airfoil
the way randomly perturbing raw coordinates could.

Each variant is evaluated directly with NeuralFoil (no .dat file needed),
so this stays fast even generating thousands of candidates -- which is
exactly what Phase 8's optimizer will do, reusing everything here.

Install what you need first (Command Prompt / terminal, not Notepad++):
    pip install aerosandbox neuralfoil numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aerosandbox as asb

# ----------------------- settings you can change -----------------------
BASELINE_DAT = "airfoils/naca2412.dat"   # starting airfoil to vary
N_VARIANTS = 200                          # how many random shapes to try
PERTURBATION_SCALE = 0.03                 # how different variants are allowed to be
                                           # (bigger = more different from baseline)

EVAL_ALPHA = 5.0        # angle of attack (deg) to score variants at
EVAL_RE = 1e6            # Reynolds number to score variants at
MODEL_SIZE = "large"

OUTPUT_CSV = "generated_airfoils.csv"
RANDOM_STATE = 42
# --------------------------------------------------------------------


def load_baseline(path):
    foil = asb.Airfoil(name="baseline", coordinates=path)
    return foil.to_kulfan_airfoil()


def make_variant(baseline_params, rng, name):
    """Nudge every weight by a small random amount to get a new, related shape."""
    lw = baseline_params["lower_weights"] + rng.normal(
        scale=PERTURBATION_SCALE, size=len(baseline_params["lower_weights"]))
    uw = baseline_params["upper_weights"] + rng.normal(
        scale=PERTURBATION_SCALE, size=len(baseline_params["upper_weights"]))
    le = baseline_params["leading_edge_weight"] + rng.normal(scale=PERTURBATION_SCALE)

    return asb.KulfanAirfoil(
        name=name,
        lower_weights=lw,
        upper_weights=uw,
        leading_edge_weight=le,
        TE_thickness=baseline_params["TE_thickness"],  # keep TE fixed, it rarely helps to vary
    )


def save_dat(foil, path):
    """Write a normal, plain-text UIUC-style .dat file -- the kind every
    other script (and Notepad++) can actually read. KulfanAirfoil's own
    .save() writes an internal binary format under the hood, not this."""
    coords = foil.coordinates
    with open(path, "w") as f:
        f.write(f"{foil.name}\n")
        for x, y in coords:
            f.write(f"  {x:.6f}  {y:.6f}\n")


def evaluate(foil, alpha, Re):
    aero = foil.get_aero_from_neuralfoil(alpha=alpha, Re=Re, model_size=MODEL_SIZE)
    cl = float(np.asarray(aero["CL"]).squeeze())
    cd = float(np.asarray(aero["CD"]).squeeze())
    return cl, cd


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    baseline = load_baseline(BASELINE_DAT)
    base_params = baseline.kulfan_parameters

    base_cl, base_cd = evaluate(baseline, EVAL_ALPHA, EVAL_RE)
    print(f"Baseline ({BASELINE_DAT}): CL={base_cl:.4f}  CD={base_cd:.5f}  "
          f"L/D={base_cl / base_cd:.1f}")

    rows = [{
        "name": "baseline", "max_thickness": baseline.max_thickness(),
        "max_camber": baseline.max_camber(), "CL": base_cl, "CD": base_cd,
        "L_over_D": base_cl / base_cd if base_cd > 1e-8 else np.nan,
    }]
    variants = {"baseline": baseline}

    print(f"Generating and evaluating {N_VARIANTS} variants...")
    for i in range(N_VARIANTS):
        name = f"variant_{i}"
        foil = make_variant(base_params, rng, name)
        try:
            cl, cd = evaluate(foil, EVAL_ALPHA, EVAL_RE)
        except Exception as e:
            continue  # skip anything NeuralFoil can't confidently evaluate
        if cd <= 0 or cd > 1 or abs(cl) > 3:
            continue  # skip physically implausible results, same as train_model.py

        variants[name] = foil
        rows.append({
            "name": name, "max_thickness": foil.max_thickness(),
            "max_camber": foil.max_camber(), "CL": cl, "CD": cd,
            "L_over_D": cl / cd if cd > 1e-8 else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("L_over_D", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} shapes to {OUTPUT_CSV}")
    print("\nTop 5 by L/D at this alpha/Re:")
    print(df.head(5).to_string(index=False))

    # Save the single best variant as a real .dat file you can inspect or
    # run through generate_dataset.py / train_model.py later
    best_name = df.iloc[0]["name"]
    if best_name != "baseline":
        save_dat(variants[best_name], "best_variant.dat")
        print(f"\nSaved best variant's geometry to best_variant.dat")

    # Quick visual: baseline vs best variant
    fig, ax = plt.subplots(figsize=(8, 3))
    for label, foil in [("baseline", baseline), ("best variant", variants[best_name])]:
        coords = foil.coordinates
        ax.plot(coords[:, 0], coords[:, 1], label=label, linewidth=1.2)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Baseline vs best variant (alpha={EVAL_ALPHA}°, Re={EVAL_RE:.0e})")
    plt.tight_layout()
    plt.savefig("baseline_vs_best.png", dpi=150)
    print("Saved comparison plot to baseline_vs_best.png")


if __name__ == "__main__":
    main()
