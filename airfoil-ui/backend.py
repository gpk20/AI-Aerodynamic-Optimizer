"""
Backend adapter for the airfoil optimizer UI (Phase 12).

The UI only ever talks to this module. Each of the three jobs below is done by
your own project code when project_hooks.py provides it, and by a clearly
labelled built-in stand-in when it doesn't:

    predict_primary     your trained CL/CD model              (stand-in: demo surrogate)
    predict_validation  your second, independent model (Ph 11) (stand-in: demo surrogate)
    optimize            your genetic-algorithm optimizer       (stand-in: built-in GA that
                                                                calls predict_primary)

Airfoil coordinates passed to hooks are always an (N, 2) float array,
chord-normalised (x from 0 to 1), Selig order: upper surface from trailing edge
to leading edge, then lower surface from leading edge to trailing edge.
"""

import glob
import math
import os
import threading
import time
import traceback
import uuid

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def _default_airfoil_dir():
    # Use whichever folder of .dat files sits next to this file: "airfoils" (the
    # name your scripts use) or "airfoil_data".
    for name in ("airfoils", "airfoil_data"):
        path = os.path.join(BASE_DIR, name)
        if os.path.isdir(path):
            return path
    return os.path.join(BASE_DIR, "airfoils")


AIRFOIL_DIR = os.environ.get("AIRFOIL_DIR") or _default_airfoil_dir()

N_GRID = 81
GRID_X = 0.5 * (1 - np.cos(np.linspace(0, np.pi, N_GRID)))
THETA = np.arccos(1 - 2 * GRID_X)
SWEEP_ALPHAS = np.arange(-4.0, 16.001, 0.5)

# --------------------------------------------------------------------------
# Hooks into the user's project
# --------------------------------------------------------------------------

_hooks = None
_hooks_error = None
try:
    import project_hooks as _hooks  # noqa: F401
except ImportError as exc:
    # Only treat "module not found" as normal; a broken import inside the file is an error.
    if getattr(exc, "name", None) != "project_hooks":
        _hooks_error = f"{type(exc).__name__}: {exc}"
except Exception as exc:  # syntax errors etc. inside project_hooks.py
    _hooks_error = f"{type(exc).__name__}: {exc}"


def _hook(name):
    fn = getattr(_hooks, name, None) if _hooks else None
    return fn if callable(fn) else None


def check_model_available():
    """A second model exists if the user hooked one, or if everything is demo."""
    return bool(_hook("predict_validation")) or not _hook("predict_primary")


def status():
    n_lib = len(_catalog())
    return {
        "sources": {
            "primary": "project" if _hook("predict_primary") else "demo",
            "check": (
                "project" if _hook("predict_validation")
                else ("demo" if check_model_available() else "none")
            ),
            "optimizer": "project" if _hook("optimize") else "built-in",
        },
        "labels": dict(getattr(_hooks, "LABELS", {}) or {}) if _hooks else {},
        "hooks_error": _hooks_error,
        "airfoil_dir": AIRFOIL_DIR,
        "library_count": n_lib,
        "default_airfoil": "naca2412",
        "sweep": {"min": float(SWEEP_ALPHAS[0]), "max": float(SWEEP_ALPHAS[-1])},
    }


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _split_surfaces(coords):
    c = np.asarray(coords, dtype=float)
    i_le = int(np.argmin(c[:, 0]))
    top = c[: i_le + 1][::-1]
    bot = c[i_le:]
    top = top[np.argsort(top[:, 0], kind="stable")]
    bot = bot[np.argsort(bot[:, 0], kind="stable")]
    if np.interp(0.5, top[:, 0], top[:, 1]) < np.interp(0.5, bot[:, 0], bot[:, 1]):
        top, bot = bot, top
    return top, bot


def to_grid(coords):
    top, bot = _split_surfaces(coords)
    yu = np.interp(GRID_X, top[:, 0], top[:, 1])
    yl = np.interp(GRID_X, bot[:, 0], bot[:, 1])
    return yu, yl


def from_grid(yu, yl):
    upper = np.column_stack([GRID_X[::-1], np.asarray(yu)[::-1]])
    lower = np.column_stack([GRID_X[1:], np.asarray(yl)[1:]])
    return np.vstack([upper, lower])


def canonical(coords):
    """Resample any airfoil onto the shared cosine grid."""
    return from_grid(*to_grid(coords))


def geometry(coords):
    yu, yl = to_grid(coords)
    t = yu - yl
    cam = 0.5 * (yu + yl)
    it = int(np.argmax(t))
    ic = int(np.argmax(np.abs(cam)))
    return {
        "max_thickness": float(t[it]),
        "max_thickness_x": float(GRID_X[it]),
        "yu_at_t": float(yu[it]),
        "yl_at_t": float(yl[it]),
        "max_camber": float(cam[ic]),
        "max_camber_x": float(GRID_X[ic]),
        "x": [round(float(v), 6) for v in GRID_X],
        "camber": [round(float(v), 6) for v in cam],
    }


def _naca4(code):
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    x = GRID_X
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if p > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    return from_grid(yc + yt * np.cos(th), yc - yt * np.cos(th))


BUILTIN = {f"naca{c}": c for c in ("0009", "0012", "2412", "2415", "4412", "4415", "6409")}


def _parse_dat(path):
    with open(path, "r", errors="ignore") as fh:
        lines = fh.read().splitlines()
    nums = []
    for ln in lines:
        parts = ln.replace(",", " ").split()
        if len(parts) != 2:
            continue
        try:
            nums.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    arr = np.array(nums, dtype=float)
    if len(arr) < 10:
        raise ValueError(f"{os.path.basename(path)} does not look like an airfoil coordinate file")
    if arr[0, 0] > 1.5 and arr[0, 1] > 1.5:  # Lednicer format: first row holds point counts
        nu, nl = int(arr[0, 0]), int(arr[0, 1])
        pts = arr[1:]
        up, lo = pts[:nu], pts[nu : nu + nl]
        arr = np.vstack([up[::-1], lo[1:]])
    xmin, xmax = arr[:, 0].min(), arr[:, 0].max()
    chord = xmax - xmin
    if chord <= 0:
        raise ValueError("zero chord length")
    arr = np.column_stack([(arr[:, 0] - xmin) / chord, arr[:, 1] / chord])
    return arr


_catalog_cache = None
_airfoil_cache = {}


def _catalog():
    global _catalog_cache
    if _catalog_cache is None:
        items = {}
        for p in sorted(glob.glob(os.path.join(AIRFOIL_DIR, "**", "*.dat"), recursive=True)):
            items[os.path.splitext(os.path.basename(p))[0].lower()] = p
        _catalog_cache = items
    return _catalog_cache


def list_airfoils(query="", limit=60):
    names = set(BUILTIN) | set(_catalog())
    q = (query or "").strip().lower()
    hits = sorted(n for n in names if q in n)
    hits.sort(key=lambda n: (not n.startswith(q), len(n), n))
    return hits[:limit]


def get_airfoil(name):
    key = (name or "").strip().lower()
    if key in _airfoil_cache:
        return _airfoil_cache[key]
    cat = _catalog()
    if key in cat:
        coords = canonical(_parse_dat(cat[key]))
    elif key in BUILTIN:
        coords = _naca4(BUILTIN[key])
    else:
        raise KeyError(name)
    _airfoil_cache[key] = coords
    return coords


# --------------------------------------------------------------------------
# Prediction (hooked or demo)
# --------------------------------------------------------------------------

def _demo_predict(coords, alpha, re, variant=0):
    """Thin-airfoil-flavoured stand-in so the UI is usable before models are hooked up."""
    yu, yl = to_grid(coords)
    t = yu - yl
    cam = 0.5 * (yu + yl)
    tmax = float(t.max())
    cam_abs = float(np.abs(cam).max())
    dz = np.gradient(cam, GRID_X)
    a0 = -(1.0 / math.pi) * float(_trapz(dz * (np.cos(THETA) - 1.0), THETA))
    a = math.radians(alpha)
    k = 1.0 + (0.77 if variant == 0 else 0.70) * tmax
    cl_lin = 2 * math.pi * (a - a0) * k * 0.92
    clmax = 1.10 + 6.0 * cam_abs + 2.5 * tmax + 0.04 * math.log10(max(re, 1e4) / 1e5)
    cl = clmax * math.tanh(cl_lin / clmax)
    re_f = (1e6 / max(re, 1e4)) ** 0.2
    cd0 = 0.0048 * re_f * (1 + 2.0 * tmax + 50 * tmax**3) * (1.0 if variant == 0 else 1.05)
    cl_des = 15.0 * cam_abs
    cd = cd0 + 0.012 * (cl - cl_des) ** 2 + 0.08 * max(0.0, abs(cl) / clmax - 0.85) ** 2
    if variant:
        cl *= 1 + 0.025 * math.sin(3 * a + 20 * tmax)
        cd *= 1 + 0.04 * math.cos(2 * a)
    return cl, cd


def predict_primary(coords, alpha, re):
    fn = _hook("predict_primary")
    cl, cd = fn(coords, alpha, re) if fn else _demo_predict(coords, alpha, re, 0)
    return float(cl), float(cd)


def predict_check(coords, alpha, re):
    fn = _hook("predict_validation")
    cl, cd = fn(coords, alpha, re) if fn else _demo_predict(coords, alpha, re, 1)
    return float(cl), float(cd)


def _ratio(cl, cd):
    return float(cl / cd) if cd and cd > 1e-6 else None


# --------------------------------------------------------------------------
# Built-in optimizer (used only when project_hooks.optimize is missing)
# --------------------------------------------------------------------------

_BUMP_LOCS = (0.08, 0.22, 0.40, 0.60, 0.80)


def _bump(loc, width=3.0):
    e = math.log(0.5) / math.log(loc)
    return np.sin(np.pi * np.power(np.clip(GRID_X, 1e-9, 1.0), e)) ** width


def _builtin_optimize(coords, alpha, re, constraints, progress):
    rng = np.random.default_rng(7)
    yu0, yl0 = to_grid(coords)
    bumps = np.array([_bump(l) for l in _BUMP_LOCS])
    k = len(_BUMP_LOCS)
    t_min, t_max, cl_min = constraints["t_min"], constraints["t_max"], constraints["cl_min"]
    pop_n, gens, amp = 40, 30, 0.03

    def build(p):
        return yu0 + p[:k] @ bumps, yl0 + p[k:] @ bumps

    def score(p):
        yu, yl = build(p)
        t = yu - yl
        tm = float(t.max())
        pen = 0.0
        if tm < t_min:
            pen += (t_min - tm) * 3000
        if tm > t_max:
            pen += (tm - t_max) * 3000
        pen += float(np.clip(-t[1:-1], 0, None).sum()) * 6000
        cl, cd = predict_primary(from_grid(yu, yl), alpha, re)
        if cl < cl_min:
            pen += (cl_min - cl) * 400
        ld = cl / cd if cd > 1e-6 else 0.0
        return ld - pen, ld

    pop = rng.normal(0, 0.008, size=(pop_n, 2 * k))
    pop[0] = 0.0
    best_p, best_s, best_ld = pop[0].copy(), -1e18, 0.0
    for g in range(gens):
        sc = np.array([score(p) for p in pop])
        order = np.argsort(-sc[:, 0])
        if sc[order[0], 0] > best_s:
            best_s, best_ld, best_p = sc[order[0], 0], sc[order[0], 1], pop[order[0]].copy()
        if progress:
            progress(g + 1, gens, best_ld)
        elite = pop[order[:2]].copy()
        children = []
        sigma = 0.006 - 0.004 * g / max(gens - 1, 1)
        while len(children) < pop_n - 2:
            a = pop[min(rng.choice(pop_n, 3), key=lambda i: -sc[i, 0])]
            b = pop[min(rng.choice(pop_n, 3), key=lambda i: -sc[i, 0])]
            w = rng.uniform(-0.3, 1.3, size=2 * k) if rng.random() < 0.8 else np.ones(2 * k)
            child = w * a + (1 - w) * b
            mask = rng.random(2 * k) < 0.25
            child = child + mask * rng.normal(0, sigma, size=2 * k)
            children.append(np.clip(child, -amp, amp))
        pop = np.vstack([elite, np.array(children)])
    yu, yl = build(best_p)
    return from_grid(yu, yl)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

JOBS = {}
_jobs_lock = threading.Lock()


def _as_coords(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 20 or not np.isfinite(arr).all():
        raise ValueError(
            "optimize() must return an (N, 2) array of x, y coordinates with at least 20 points; "
            f"it returned shape {arr.shape}"
        )
    return canonical(arr)


def _point(coords, alpha, re):
    cl, cd = predict_primary(coords, alpha, re)
    out = {"cl": cl, "cd": cd, "ld": _ratio(cl, cd)}
    if check_model_available():
        cl2, cd2 = predict_check(coords, alpha, re)
        out["check"] = {"cl": cl2, "cd": cd2, "ld": _ratio(cl2, cd2)}
    return out


def _side(coords, alpha, re):
    return {
        "coords": [[round(float(x), 6), round(float(y), 6)] for x, y in coords],
        "geometry": geometry(coords),
        "at_design": _point(coords, alpha, re),
    }


def _build_result(base, opt, alpha, re):
    sweep = {"alpha": [float(a) for a in SWEEP_ALPHAS], "base": {"cl": [], "cd": [], "ld": []},
             "opt": {"cl": [], "cd": [], "ld": []}}
    for a in SWEEP_ALPHAS:
        for key, c in (("base", base), ("opt", opt)):
            cl, cd = predict_primary(c, float(a), re)
            sweep[key]["cl"].append(cl)
            sweep[key]["cd"].append(cd)
            sweep[key]["ld"].append(_ratio(cl, cd))
    better = sum(
        1 for b, o in zip(sweep["base"]["ld"], sweep["opt"]["ld"]) if b is not None and o is not None and o > b
    )
    return {
        "alpha": alpha,
        "re": re,
        "baseline": _side(base, alpha, re),
        "optimized": _side(opt, alpha, re),
        "sweep": sweep,
        "angles_better": better,
        "angles_total": len(SWEEP_ALPHAS),
    }


def _run_job(job):
    p = job["params"]
    try:
        base = get_airfoil(p["name"])

        def progress(gen, total, best=None):
            job["generation"] = int(gen)
            job["total"] = int(total)
            job["progress"] = float(gen) / total if total else 0.0
            job["history"].append(None if best is None else float(best))

        fn = _hook("optimize")
        if fn:
            opt = fn(base, p["alpha"], p["re"], dict(p["constraints"]), progress)
        else:
            opt = _builtin_optimize(base, p["alpha"], p["re"], p["constraints"], progress)
        opt = _as_coords(opt)
        job["result"] = _build_result(base, opt, p["alpha"], p["re"])
        job["progress"] = 1.0
        job["status"] = "done"
    except Exception as exc:  # surfaced to the UI
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["finished"] = time.time()


def start_job(params):
    jid = uuid.uuid4().hex[:8]
    job = {
        "id": jid, "status": "running", "progress": 0.0, "generation": 0, "total": 0,
        "history": [], "params": params, "result": None, "error": None,
        "created": time.time(), "finished": None,
    }
    with _jobs_lock:
        JOBS[jid] = job
        for old in sorted(JOBS, key=lambda k: JOBS[k]["created"])[:-40]:
            JOBS.pop(old, None)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def job_view(job):
    return {k: job[k] for k in ("id", "status", "progress", "generation", "total", "history", "error", "result", "params", "finished")}


def export_text(job, which, fmt):
    side = job["result"][which]
    name = job["params"]["name"]
    label = f"{name} {which}"
    if fmt == "csv":
        body = "x,y\n" + "\n".join(f"{x:.6f},{y:.6f}" for x, y in side["coords"])
        return body + "\n", f"{name}_{which}.csv", "text/csv"
    body = label + "\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in side["coords"])
    return body + "\n", f"{name}_{which}.dat", "text/plain"
