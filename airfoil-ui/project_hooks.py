"""
Connects your Phase 1-11 code to the interface.

What each part uses, taken from your own scripts:

  predict_primary     NeuralFoil on the exact shape. This is what your optimizer
                      (optimize_airfoil.py) used to score every candidate, so the
                      plots show the improvement the way the optimizer saw it.
  predict_validation  The gradient-boosting model from train_model.py /
                      validate_optimized.py: trained on airfoil_dataset.csv and
                      fed the four shape descriptors + angle + log10(Re).
  optimize            Your genetic algorithm from optimize_airfoil.py, with the
                      angle, Reynolds number and limits taken from the page.

Needs: pip install aerosandbox neuralfoil numpy pandas scikit-learn

For the check model, copy airfoil_dataset.csv into this folder (or set the
AIRFOIL_DATASET environment variable to its full path). Without it the check
model shows as "Not connected" and everything else still works.
"""

import os
import numpy as np
import pandas as pd
import aerosandbox as asb
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))

# Shown in the title block at the bottom of the page.
LABELS = {
    "primary": "NeuralFoil",
    "check": "Gradient boosting on your dataset",
    "optimizer": "Your genetic algorithm",
}

MODEL_SIZE = "large"            # NeuralFoil size, same as your scripts

# Genetic algorithm settings, copied from optimize_airfoil.py
POP_SIZE = 40
N_GENERATIONS = 30
ELITE_FRACTION = 0.2
MUTATION_SCALE = 0.02
INIT_SCALE = 0.04
MIN_CAMBER = 0.0
MAX_CAMBER = 0.06
RANDOM_STATE = 42
N_WEIGHTS = 8
PENALTY = -1e6

# Check-model settings, copied from train_model.py / validate_optimized.py
FEATURE_COLUMNS = [
    "max_thickness", "max_thickness_loc",
    "max_camber", "max_camber_loc",
    "alpha", "log_reynolds",
]


# --------------------------------------------------------------------------
# Primary prediction: NeuralFoil on the shape itself
# --------------------------------------------------------------------------

def _foil(coords):
    return asb.Airfoil(name="shape", coordinates=np.asarray(coords, dtype=float))


def _neuralfoil(foil, alpha, re):
    aero = foil.get_aero_from_neuralfoil(alpha=alpha, Re=re, model_size=MODEL_SIZE)
    return float(np.asarray(aero["CL"]).squeeze()), float(np.asarray(aero["CD"]).squeeze())


def predict_primary(coords, alpha, re):
    """Return (CL, CD) from NeuralFoil."""
    return _neuralfoil(_foil(coords), alpha, re)


# --------------------------------------------------------------------------
# Check model: gradient boosting trained on your dataset (same as Phase 11)
# --------------------------------------------------------------------------

def _find_dataset():
    candidates = [os.environ.get("AIRFOIL_DATASET"), os.path.join(HERE, "airfoil_dataset.csv")]
    return next((p for p in candidates if p and os.path.isfile(p)), None)


def _shape_descriptors(foil):
    """Same four numbers generate_dataset.py computed for every training airfoil."""
    xs = np.linspace(0.001, 0.999, 300)
    thickness = foil.local_thickness(xs)
    camber = foil.local_camber(xs)
    return {
        "max_thickness": float(thickness.max()),
        "max_thickness_loc": float(xs[np.argmax(thickness)]),
        "max_camber": float(camber.max()),
        "max_camber_loc": float(xs[np.argmax(camber)]),
    }


def _train_check_models(path):
    df = pd.read_csv(path)
    df["log_reynolds"] = np.log10(df["reynolds"])
    cols = FEATURE_COLUMNS + ["CL", "CD"]
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=cols)
    df = df[(df["max_thickness"] > 0) & (df["max_thickness"] <= 0.6) &
            (df["max_camber"].abs() <= 0.3) & (df["CL"].abs() <= 3) &
            (df["CD"] > 0) & (df["CD"] <= 1)]
    models = {}
    for target in ("CL", "CD"):
        m = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        m.fit(df[FEATURE_COLUMNS], df[target])
        models[target] = m
    return models


_dataset = _find_dataset()
if _dataset:
    print(f"[project_hooks] Training the check model on {_dataset} ...")
    _check = _train_check_models(_dataset)
    print("[project_hooks] Check model ready.")

    def predict_validation(coords, alpha, re):
        """Return (CL, CD) from the gradient-boosting check model."""
        row = _shape_descriptors(_foil(coords))
        row.update(alpha=float(alpha), log_reynolds=float(np.log10(re)))
        x = pd.DataFrame([row])[FEATURE_COLUMNS]
        return float(_check["CL"].predict(x)[0]), float(_check["CD"].predict(x)[0])
else:
    print("[project_hooks] airfoil_dataset.csv not found next to project_hooks.py, "
          "so the check model is not connected.")


# --------------------------------------------------------------------------
# Optimizer: your genetic algorithm, driven by the page's settings
# --------------------------------------------------------------------------

def _vector_to_airfoil(vector, te_thickness, name="candidate"):
    return asb.KulfanAirfoil(
        name=name,
        lower_weights=vector[:N_WEIGHTS],
        upper_weights=vector[N_WEIGHTS:2 * N_WEIGHTS],
        leading_edge_weight=vector[2 * N_WEIGHTS],
        TE_thickness=te_thickness,
    )


def _fitness(vector, te_thickness, alpha, re, limits):
    """L/D at the design point, heavily penalized when a limit is broken."""
    try:
        foil = _vector_to_airfoil(vector, te_thickness)
        thickness = foil.max_thickness()
        camber = foil.max_camber()
        if not (limits["t_min"] <= thickness <= limits["t_max"]):
            return PENALTY
        if not (MIN_CAMBER <= camber <= MAX_CAMBER):
            return PENALTY
        cl, cd = _neuralfoil(foil, alpha, re)
        if cd <= 0 or cd > 1 or abs(cl) > 3 or not np.isfinite(cl) or not np.isfinite(cd):
            return PENALTY
        if cl < limits["cl_min"]:
            return PENALTY
        return cl / cd
    except Exception:
        return PENALTY


def optimize(coords, alpha, re, constraints, progress):
    """Evolve the shape and return the best one as an (N, 2) coordinate array."""
    rng = np.random.default_rng(RANDOM_STATE)
    kulfan = _foil(coords).to_kulfan_airfoil()
    p = kulfan.kulfan_parameters
    base = np.concatenate([p["lower_weights"], p["upper_weights"], [p["leading_edge_weight"]]])
    te = p["TE_thickness"]

    def fit(v):
        return _fitness(v, te, alpha, re, constraints)

    population = [base.copy()] + [base + rng.normal(scale=INIT_SCALE, size=base.shape)
                                  for _ in range(POP_SIZE - 1)]
    n_elite = max(1, int(POP_SIZE * ELITE_FRACTION))
    best_vec, best_fit = base.copy(), fit(base)

    for gen in range(N_GENERATIONS):
        scored = sorted(((v, fit(v)) for v in population), key=lambda r: r[1], reverse=True)
        if scored[0][1] > best_fit:
            best_vec, best_fit = scored[0][0].copy(), scored[0][1]
        if progress:
            progress(gen + 1, N_GENERATIONS, best_fit if best_fit > PENALTY else None)

        elites = [r[0] for r in scored[:n_elite]]
        nxt = [e.copy() for e in elites]
        while len(nxt) < POP_SIZE:
            a, b = elites[rng.integers(n_elite)], elites[rng.integers(n_elite)]
            mix = rng.uniform(0, 1, size=a.shape)
            nxt.append(mix * a + (1 - mix) * b + rng.normal(scale=MUTATION_SCALE, size=a.shape))
        population = nxt

    if best_fit <= PENALTY:
        raise ValueError(
            "No shape met all the limits, including the 0 to 6% camber limit. "
            "Widen the thickness range or lower the minimum CL, then run again."
        )
    return _vector_to_airfoil(best_vec, te, name="optimized").coordinates
