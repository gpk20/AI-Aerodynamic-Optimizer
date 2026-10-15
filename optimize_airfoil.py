"""
optimize_airfoil.py
---------------------
Phase 8 of the AI Aerodynamic Optimizer project.

A genetic algorithm (GA) that evolves airfoil shapes toward higher L/D,
instead of Phase 7's purely random tries. The idea, in plain terms:

  1. Start with a POPULATION of candidate shapes (random variants of a
     baseline airfoil, same as Phase 7).
  2. Score every shape's L/D (its "fitness").
  3. Keep the best few shapes unchanged ("elitism" -- never lose your best).
  4. Breed new shapes by blending pairs of good shapes together
     ("crossover"), then nudge them randomly ("mutation").
  5. Repeat for many generations. Each generation should trend better,
     because you keep breeding from what already worked.

Every shape is represented the same compact way as Phase 7 (a Kulfan/CST
parameter vector of 17 numbers) and evaluated directly with NeuralFoil --
no .dat files written during the search, so this can try thousands of
shapes quickly.

Install what you need first (Command Prompt / terminal, not Notepad++):
    pip install aerosandbox neuralfoil numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aerosandbox as asb

# ----------------------- settings you can change -----------------------
BASELINE_DAT = "airfoils/naca2412.dat"

EVAL_ALPHA = 5.0     # angle of attack (deg) the optimizer scores shapes at
EVAL_RE = 1e6         # Reynolds number the optimizer scores shapes at
MODEL_SIZE = "large"

POP_SIZE = 40
N_GENERATIONS = 30
ELITE_FRACTION = 0.2    # top fraction of each generation carried over unchanged
MUTATION_SCALE = 0.02   # how much a child's genes get nudged after breeding
INIT_SCALE = 0.04        # how different the starting population is from baseline

# --- Phase 9: constraints, so the optimizer can't wander into unrealistic
# shapes just because they trick the model into predicting a good L/D ---
MIN_THICKNESS = 0.08     # 8% chord -- thinner gets structurally fragile
MAX_THICKNESS = 0.18     # 18% chord -- thicker is draggy and unusual
MIN_CAMBER = 0.0
MAX_CAMBER = 0.06        # 6% chord camber is already a lot for most wings
MIN_CL = 0.6             # must still produce meaningful lift at the design point

RANDOM_STATE = 42
# --------------------------------------------------------------------

N_WEIGHTS = 8  # per side, fixed by the Kulfan parameterization used in Phase 7


def save_dat(foil, path):
    """Write a normal, plain-text UIUC-style .dat file -- the kind every
    other script (and Notepad++) can actually read. KulfanAirfoil's own
    .save() writes an internal binary format under the hood, not this."""
    coords = foil.coordinates
    with open(path, "w") as f:
        f.write(f"{foil.name}\n")
        for x, y in coords:
            f.write(f"  {x:.6f}  {y:.6f}\n")


def load_baseline():
    foil = asb.Airfoil(name="baseline", coordinates=BASELINE_DAT).to_kulfan_airfoil()
    p = foil.kulfan_parameters
    vector = np.concatenate([p["lower_weights"], p["upper_weights"], [p["leading_edge_weight"]]])
    return foil, vector, p["TE_thickness"]


def vector_to_airfoil(vector, te_thickness, name="candidate"):
    return asb.KulfanAirfoil(
        name=name,
        lower_weights=vector[:N_WEIGHTS],
        upper_weights=vector[N_WEIGHTS:2 * N_WEIGHTS],
        leading_edge_weight=vector[2 * N_WEIGHTS],
        TE_thickness=te_thickness,
    )


PENALTY = -1e6  # fitness assigned to any shape that breaks a constraint


def fitness(vector, te_thickness):
    """L/D at the chosen design point -- but only for shapes that satisfy
    Phase 9's constraints. Anything that breaks a constraint (too thick,
    too thin, not enough camber, not enough lift, or a shape NeuralFoil
    can't confidently evaluate) gets heavily penalized, so the GA breeds
    away from that region instead of exploiting it."""
    try:
        foil = vector_to_airfoil(vector, te_thickness)

        # geometry constraints -- checked first, cheaply, before running NeuralFoil
        thickness = foil.max_thickness()
        camber = foil.max_camber()
        if not (MIN_THICKNESS <= thickness <= MAX_THICKNESS):
            return PENALTY, np.nan, np.nan
        if not (MIN_CAMBER <= camber <= MAX_CAMBER):
            return PENALTY, np.nan, np.nan

        aero = foil.get_aero_from_neuralfoil(alpha=EVAL_ALPHA, Re=EVAL_RE, model_size=MODEL_SIZE)
        cl = float(np.asarray(aero["CL"]).squeeze())
        cd = float(np.asarray(aero["CD"]).squeeze())
        if cd <= 0 or cd > 1 or abs(cl) > 3 or not np.isfinite(cl) or not np.isfinite(cd):
            return PENALTY, cl, cd

        # aerodynamic constraint
        if cl < MIN_CL:
            return PENALTY, cl, cd

        return cl / cd, cl, cd
    except Exception:
        return PENALTY, np.nan, np.nan


def crossover(parent_a, parent_b, rng):
    """Blend two parents gene-by-gene with a random mix ratio."""
    mix = rng.uniform(0, 1, size=parent_a.shape)
    return mix * parent_a + (1 - mix) * parent_b


def mutate(vector, rng, scale):
    return vector + rng.normal(scale=scale, size=vector.shape)


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    baseline_foil, baseline_vector, te_thickness = load_baseline()

    base_ld, base_cl, base_cd = fitness(baseline_vector, te_thickness)
    print(f"Baseline ({BASELINE_DAT}): CL={base_cl:.4f}  CD={base_cd:.5f}  L/D={base_ld:.1f}")

    # --- initial population: baseline + random variants around it ---
    population = [baseline_vector.copy()]
    for _ in range(POP_SIZE - 1):
        population.append(mutate(baseline_vector, rng, INIT_SCALE))

    n_elite = max(1, int(POP_SIZE * ELITE_FRACTION))
    history = []
    best_vector, best_ld = baseline_vector.copy(), base_ld

    for gen in range(N_GENERATIONS):
        scored = [(vec, *fitness(vec, te_thickness)) for vec in population]
        scored.sort(key=lambda row: row[1], reverse=True)  # best L/D first

        gen_best_vec, gen_best_ld = scored[0][0], scored[0][1]
        if gen_best_ld > best_ld:
            best_vector, best_ld = gen_best_vec.copy(), gen_best_ld

        valid_ld = [s[1] for s in scored if s[1] > PENALTY]
        mean_ld = np.mean(valid_ld) if valid_ld else float("nan")
        n_valid = len(valid_ld)
        history.append({"generation": gen, "best_L_over_D": gen_best_ld,
                         "mean_L_over_D": mean_ld, "n_valid": n_valid})
        print(f"Gen {gen:3d}  best L/D = {gen_best_ld:7.1f}   mean L/D = {mean_ld:7.1f}"
              f"   ({n_valid}/{POP_SIZE} satisfied constraints)")

        # --- next generation: elitism + breed the rest ---
        elites = [row[0] for row in scored[:n_elite]]
        next_population = [e.copy() for e in elites]
        while len(next_population) < POP_SIZE:
            parent_a, parent_b = elites[rng.integers(n_elite)], elites[rng.integers(n_elite)]
            child = mutate(crossover(parent_a, parent_b, rng), rng, MUTATION_SCALE)
            next_population.append(child)
        population = next_population

    print(f"\nBest L/D found: {best_ld:.1f}  (baseline was {base_ld:.1f}, "
          f"+{100 * (best_ld / base_ld - 1):.1f}%)")

    # --- save results ---
    best_foil = vector_to_airfoil(best_vector, te_thickness, name="optimized")
    save_dat(best_foil, "optimized_airfoil.dat")

    pd.DataFrame(history).to_csv("optimization_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    hist_df = pd.DataFrame(history)
    ax.plot(hist_df["generation"], hist_df["best_L_over_D"], label="best in generation")
    ax.plot(hist_df["generation"], hist_df["mean_L_over_D"], label="population average", alpha=0.6)
    ax.axhline(base_ld, color="gray", linestyle="--", linewidth=1, label="baseline")
    ax.set_xlabel("Generation")
    ax.set_ylabel("L/D")
    ax.set_title("Optimization progress")
    ax.legend()
    plt.tight_layout()
    plt.savefig("optimization_progress.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 3))
    for label, foil in [("baseline", baseline_foil), ("optimized", best_foil)]:
        coords = foil.coordinates
        ax.plot(coords[:, 0], coords[:, 1], label=label, linewidth=1.2)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Baseline vs optimized (alpha={EVAL_ALPHA}°, Re={EVAL_RE:.0e})")
    plt.tight_layout()
    plt.savefig("baseline_vs_optimized.png", dpi=150)

    print("\nSaved: optimized_airfoil.dat, optimization_history.csv, "
          "optimization_progress.png, baseline_vs_optimized.png")


if __name__ == "__main__":
    main()
