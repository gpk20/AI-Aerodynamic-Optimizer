"""Run with:  python app.py   then open http://127.0.0.1:5000"""

from flask import Flask, Response, jsonify, request, send_from_directory

import backend

app = Flask(__name__, static_folder="static", static_url_path="/static")


def _num(body, key, lo, hi, label):
    try:
        value = float(body[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{label} must be a number.")
    if not (lo <= value <= hi):
        raise ValueError(f"{label} must be between {lo:g} and {hi:g}.")
    return value


@app.errorhandler(ValueError)
def _bad_value(exc):
    return jsonify(error=str(exc)), 400


@app.errorhandler(KeyError)
def _missing(exc):
    name = exc.args[0] if exc.args else ""
    return jsonify(error=f"No airfoil named '{name}'. Pick one from the list."), 404


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/status")
def status():
    return jsonify(backend.status())


@app.get("/api/airfoils")
def airfoils():
    return jsonify(names=backend.list_airfoils(request.args.get("q", "")))


@app.get("/api/airfoil/<name>")
def airfoil(name):
    coords = backend.get_airfoil(name)
    return jsonify(
        name=name.lower(),
        coords=[[round(float(x), 6), round(float(y), 6)] for x, y in coords],
        geometry=backend.geometry(coords),
    )


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    alpha = _num(body, "alpha", -10, 20, "Angle of attack")
    re = _num(body, "re", 5e4, 1e7, "Reynolds number")
    coords = backend.get_airfoil(body.get("name", ""))
    cl, cd = backend.predict_primary(coords, alpha, re)
    return jsonify(cl=cl, cd=cd, ld=(cl / cd if cd > 1e-6 else None))


@app.post("/api/optimize")
def optimize():
    body = request.get_json(silent=True) or {}
    alpha = _num(body, "alpha", -10, 20, "Angle of attack")
    re = _num(body, "re", 5e4, 1e7, "Reynolds number")
    cons = body.get("constraints") or {}
    t_min = _num(cons, "t_min", 0.02, 0.4, "Minimum thickness")
    t_max = _num(cons, "t_max", 0.02, 0.4, "Maximum thickness")
    cl_min = _num(cons, "cl_min", -2, 3, "Minimum CL")
    if t_min > t_max:
        raise ValueError("Minimum thickness must not be larger than maximum thickness.")
    backend.get_airfoil(body.get("name", ""))  # 404 early for an unknown airfoil
    job = backend.start_job({
        "name": body["name"].lower(), "alpha": alpha, "re": re,
        "constraints": {"t_min": t_min, "t_max": t_max, "cl_min": cl_min},
    })
    return jsonify(id=job["id"]), 202


@app.get("/api/jobs/<jid>")
def job(jid):
    j = backend.JOBS.get(jid)
    if j is None:
        return jsonify(error="That run is no longer available. Run the optimizer again."), 404
    return jsonify(backend.job_view(j))


@app.get("/api/jobs/<jid>/export")
def export(jid):
    j = backend.JOBS.get(jid)
    if j is None or j["status"] != "done":
        return jsonify(error="No finished run to export."), 404
    which = "baseline" if request.args.get("which") == "baseline" else "optimized"
    fmt = "csv" if request.args.get("format") == "csv" else "dat"
    body, filename, mime = backend.export_text(j, which, fmt)
    return Response(body, mimetype=mime, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
