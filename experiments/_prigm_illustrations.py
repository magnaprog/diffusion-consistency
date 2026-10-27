"""Verify the released imputation examples and their figure data.

After the original-study and extension checks pass, compare the two example
windows with the public observations, recompute their eight primary energy
scores, and check the recorded figure coordinates. Decode the Gaussian
excerpt directly from the public Gaussian ensemble. The complete neural
ensembles are unavailable here: their recorded hashes connect the excerpts
to the earlier audit that decoded the full saved predictions.

The module also checks the fixed-final coefficient plot against its published
score records. It uses no producer helpers and performs no plotting, fitting
or neural prediction replay. Matching a recorded hash and recomputing an
available quantity are separate parts of this verification.
"""
from pathlib import Path
import math

import numpy as np


PINS = {
    "evidence/imputation/sample-excerpt.npz": "f3b270d52ea52088fa84414fcb77b6954624913ed474618be245de417034a1cc",
    "evidence/imputation/MANIFEST.json": "4416bd1c76aa849de65595e445edc09d5230f2e0bfd2afc75c30c888697c3c66",
    "figures/fixed_final/fig_fixed_final.manifest.json": "db2436baa4ae0229a870eafef05c867ab1e77d94426c5d0b65fd6111cc721f44",
    "figures/imputation/case_120/fig_imputation.manifest.json": "5d2fbf52337f17d0722cfe19d6c1c975ae1ac4746832e50bc1cbb3f842e1b1f1",
    "figures/imputation/case_360/fig_imputation.manifest.json": "d5823fe76c1deecde13210b32f87ada619716079ac36d12ba1a3a2bbcf3d236f",
    "figures/imputation/detail/fig_imputation.manifest.json": "29802ad07eebd26ab31262441e7441aa78f5e3492e54bd4677a76d601c4d4a40",
}
METHODS = ("D", "O-S", "G5", "full_flank_gaussian")
INDICES = (120, 360)
STEPS = {"D": 9560, "O-S": 29636, "G5": 29636}
CHANNELS = ["mu_x", "mu_y", "mu_z", "m", "v_s"]
UNITS = ["nT", "nT", "nT", "nT", "nT^2"]
SELECTION = {"rule": "midpoints_of_two_equal_ordered_TEST_partitions",
             "zero_based_indices": list(INDICES), "original_window_count": 480,
             "neural_seed": 7, "displayed_draws": [0, 1, 2], "after_test_exposure": True}
GAUSSIAN_PROTOCOL = "8456fdae42a086fa9cb11dbc4bc4da9d54d5a3fe233aff6ed6d4bb30d25d0090"
GAUSSIAN_ROOT = "evidence/deadline/gaussian/test/lambda_0.1_rho_0"
FIXED_SCOPE = (
    "Raw month-balanced TEST metrics at the predetermined final update 29636. "
    "Historical TEST exposure; descriptive coefficient response, no coefficient selection. "
    "Three fixed seed traces and their arithmetic mean, not uncertainty intervals. "
    "Straight segments connect observed seed values; no fitted curve, uncertainty bands, "
    "corrected results, or VAL-selected results.")
IMPUTATION_SCOPE = (
    "Raw draws, no pooled seeds, smoothing, sampling or clipping. Nominal marginal bands "
    "are not a calibration claim. Examples selected after TEST exposure.")


def _pin(g, paper, name):
    g.agree(g.digest(g.local(paper, name)), PINS[name], "independent illustration pin " + name)


def _exact(g, reported, actual, label):
    """Exact coordinates, accepting JSON lists for Python tuples, never NaN/bools."""
    # agree is tolerant for floats. Use it for structural/type diagnostics first,
    # then insist on exact numerical copies (only derived energies use tolerance).
    g.agree(reported, actual, label)
    if isinstance(actual, dict):
        for key in actual:
            _exact(g, reported[key], actual[key], label + "." + key)
    elif isinstance(actual, (list, tuple)):
        for i, (left, right) in enumerate(zip(reported, actual)):
            _exact(g, left, right, f"{label}[{i}]")
    else:
        g.require(reported == actual, label + ": exact value differs")


def _energy(samples, truth, mean, scale):
    """Energy-score estimator in 60 standardized coordinates using 39 draws.

    The pairwise term averages off-diagonal draw pairs.
    """
    draws = ((samples - mean[:5]) / scale[:5]).reshape(39, 60)
    target = ((truth - mean[:5]) / scale[:5]).reshape(60)
    target_term = np.linalg.norm(draws - target, axis=1).mean()
    distances = np.linalg.norm(draws[:, None] - draws[None, :], axis=2)
    return float(target_term - distances.sum() / (2 * 39 * 38))


def _coordinates(truth, samples):
    x = (np.arange(48) * 5 - 90).tolist()
    ordered = np.sort(samples, axis=1)
    coords = {"time_bin_start_minutes": x, "gap_minutes": [0, 60],
              "context_intervals_half_open_minutes": [[-90, 0], [60, 150]],
              "truth_channel_indices": [0, 1, 2, 3, 4],
              "ranks": {"draw_axis_in_excerpt": 2, "draw_count": 39, "zero_based_low": 1,
                        "zero_based_median": 19, "zero_based_high": 37,
                        "displayed_draws": [0, 1, 2], "interpolation": "none"}}
    for key, start, end in (("observed_left", 0, 18), ("withheld_truth", 18, 30),
                            ("observed_right", 30, 48)):
        coords[key] = {"x": x[start:end], "y": truth[start:end, :5].tolist()}
    coords["methods"] = {
        method: {"x": x[18:30], "low": ordered[i, 1].tolist(),
                 "median": ordered[i, 19].tolist(), "high": ordered[i, 37].tolist(),
                 "draws": samples[i, :3].tolist()} for i, method in enumerate(METHODS)}
    return coords


def _figure(g, paper, directory, name, inputs, expected):
    root = g.local(paper, directory)
    filename = name + ".manifest.json"
    _pin(g, paper, directory + "/" + filename)
    manifest = g.read_json(g.local(root, filename))
    g.private_check(manifest)
    source = name + ".py"
    source_hash = g.digest(g.local(g.PAPER, source))
    g.agree(g.digest(g.local(paper, source)), source_hash, "supplied figure source " + source)
    header = {"schema_version": 1, "figure": name,
              "source": {"path": source, "sha256": source_hash},
              "style": {"path": "prigm.mplstyle",
                        "sha256": g.digest(g.local(paper, "prigm.mplstyle"))},
              "inputs": inputs, **expected}
    g.require(set(manifest) == {*header, "matplotlib_version", "output_sha256"},
              name + ": figure fields differ")
    _exact(g, {k: manifest[k] for k in header}, header, name)
    g.require(isinstance(manifest["matplotlib_version"], str) and bool(manifest["matplotlib_version"]),
              "missing matplotlib version")
    outputs = {name + ".pdf", name + ".png"}
    g.require(set(manifest["output_sha256"]) == outputs, "figure output inventory differs")
    g.deadline_inventory(root, outputs | {filename})
    for output, sha in manifest["output_sha256"].items():
        path = g.local(root, output)
        g.agree(g.digest(path), sha, "figure output hash " + output)
        g.require(path.read_bytes().startswith(b"%PDF-" if output.endswith(".pdf") else b"\x89PNG\r\n\x1a\n"),
                  "figure output header differs")
    return manifest


def validate_illustrations(paper_dir, observations):
    """Check illustration records after the original-study and extension checks.

    observations is a tuple containing the physical array with shape
    (480, 48, 6), its ordered window IDs, the normalizer and the array
    identity record. Verify these values against the public observation file,
    then check the released two-window excerpt and its numerical annotations.

    Return the checked case IDs, primary scores and figure-record identities.
    The full neural ensembles are unavailable: their hashes and selected
    checkpoints are checked, while the public Gaussian excerpt is decoded.
    """
    from experiments import verify_prigm_paper as g

    paper = Path(paper_dir)
    excerpt_name = "evidence/imputation/sample-excerpt.npz"
    manifest_name = "evidence/imputation/MANIFEST.json"
    root = g.local(paper, "evidence/imputation")
    g.deadline_inventory(root, {"sample-excerpt.npz", "MANIFEST.json"})
    for name in PINS:
        _pin(g, paper, name)
    record = g.read_json(g.local(paper, manifest_name))
    g.private_check(record)
    z, ids, nrm, identity = observations
    g.require(np.asarray(z).shape == (480, 48, 6) and np.isfinite(z).all()
              and len(ids) == len(set(ids)) == 480, "illustration observations shape/IDs differ")
    with np.load(g.local(paper, excerpt_name), allow_pickle=False) as blob:
        g.require(len(blob.files) == 4 and set(blob.files) == {"truth", "samples", "ids", "methods"},
                  "excerpt NPZ fields differ")
        data = {key: blob[key] for key in blob.files}
    for key, shape in (("truth", (2, 48, 6)), ("samples", (4, 2, 39, 12, 5))):
        a = data[key]
        g.require(a.shape == shape and a.dtype == np.float64 and np.isfinite(a).all(),
                  "excerpt shape/dtype/finite differs: " + key)
    for key, values in (("ids", [ids[i] for i in INDICES]), ("methods", list(METHODS))):
        g.require(data[key].dtype.kind == "U" and data[key].shape == (len(values),),
                  "excerpt Unicode vector differs: " + key)
        g.agree(data[key].tolist(), values, "excerpt " + key)
    g.require(np.array_equal(data["truth"], np.asarray(z)[list(INDICES)]), "excerpt physical truth differs")

    real = g.read_json(g.local(paper, "evidence/real/MANIFEST.json"))
    deadline = g.read_json(g.local(paper, "evidence/deadline/MANIFEST.json"))
    comparison = g.read_json(g.local(paper, "evidence/real/comparison.json"))
    scores = g.read_json(g.local(paper, "evidence/real/scores.json"))
    gaussian = g.read_json(g.local(paper, "evidence/deadline/gaussian/test/SCORES.json"))
    for obj in (comparison, scores, gaussian):
        g.agree(obj["ids"], list(ids), "full ordered score IDs")
    for obj in (comparison, gaussian):
        g.agree(obj["arrays"], identity, "excerpt score array identity")
    for obj in (comparison, scores):
        g.agree(obj["protocol_sha256"], g.DEADLINE_PROTOCOL, "excerpt original protocol")
    # Bind the passed physical arrays and normalization to the public observation
    # bytes, rather than accepting a caller-supplied identity with different data.
    obs_path = g.local(paper, "evidence/observations/test.npz")
    g.agree(g.digest(obs_path), identity["sha256"], "physical observation hash")
    with np.load(obs_path, allow_pickle=False) as blob:
        g.require(np.array_equal(blob["z"], z), "full physical observation arrays differ")
        g.agree(blob["ids"].tolist(), list(ids), "physical observation ID order")
        for key in ("mean", "scale"):
            g.require(np.array_equal(blob[key], getattr(nrm, key)), "normalization " + key + " differs")
        for key in ("b_ref_sq", "n_bins"):
            g.agree(blob[key].item(), getattr(nrm, key), "normalization " + key)
    g.require(np.asarray(nrm.mean).shape == np.asarray(nrm.scale).shape == (6,)
              and np.isfinite(nrm.mean).all() and np.isfinite(nrm.scale).all()
              and (nrm.scale > 0).all(), "invalid excerpt normalization")

    core = {p.relative_to(g.REPO).as_posix(): g.execution_source_hash(p.relative_to(g.REPO).as_posix())
            for p in (g.REPO / "prigm").glob("*.py")}
    sources = {**core, "analysis/export_prigm_sample_excerpt.py":
               g.documentation.source_hash(g.REPO, "analysis/export_prigm_sample_excerpt.py")}
    public_names = {"evidence/real/MANIFEST.json", "evidence/deadline/MANIFEST.json",
                    "evidence/real/comparison.json", "evidence/real/scores.json",
                    "evidence/deadline/gaussian/test/SCORES.json", "evidence/observations/test.npz",
                    *(GAUSSIAN_ROOT + "/gen_00000000/" + f for f in ("COMMIT.json", "ensemble.npz"))}
    inputs = {name: g.digest(g.local(paper, name)) for name in public_names}
    for filename in ("comparison.json", "scores.json"):
        g.agree(inputs["evidence/real/" + filename], real["output_sha256"][filename], "original public input")
    for name in public_names:
        if name.startswith("evidence/deadline/") and not name.endswith("MANIFEST.json"):
            g.agree(inputs[name], deadline["files"][name.removeprefix("evidence/deadline/")], "deadline public input")

    commit = g.read_json(g.local(paper, GAUSSIAN_ROOT + "/gen_00000000/COMMIT.json"))
    control = {"family": "full_flank_linear_gaussian", "jitter": 1e-6,
               "joint_gaussian_channels": 5, "lambda": 0.1, "rho": 0.0}
    fit_select = g.DEADLINE_ORIGINALS["gaussian/FIT_SELECT.json"]
    fit_hash = deadline["files"]["gaussian/FIT.npz"]
    for key, value in {"schema_version": 2, "generation": 0, "procedure": "gaussian",
                       "protocol_sha256": GAUSSIAN_PROTOCOL, "source_protocol_sha256": g.DEADLINE_PROTOCOL,
                       "fit_select_sha256": fit_select, "control_fit_sha256": fit_hash,
                       "control": control, "arrays": identity, "ids_sha256": identity["ids_sha256"],
                       "n_generated": 6, "core_source_hashes": core,
                       "files": {"ensemble.npz": inputs[GAUSSIAN_ROOT + "/gen_00000000/ensemble.npz"]}}.items():
        g.agree(commit[key], value, "Gaussian COMMIT " + key)
    g.agree(commit["noise"]["seed"], 13, "Gaussian noise seed")
    g.agree(commit["noise"]["n_draws"], 39, "Gaussian draw count")
    for key, value in {"extension_protocol_sha256": GAUSSIAN_PROTOCOL,
                       "source_protocol_sha256": g.DEADLINE_PROTOCOL, "fit_select_sha256": fit_select,
                       "fit_payload_sha256": fit_hash, "n": 480, "n_draws": 39}.items():
        g.agree(gaussian[key], value, "Gaussian score " + key)
    selected = g.read_json(g.local(paper, "evidence/deadline/gaussian/SELECTION.json"))
    for obj in (selected, gaussian):
        g.agree(obj["selected"]["raw"]["lambda"], 0.1, "Gaussian raw selected lambda")
        g.agree(obj["selected"]["raw"]["rho"], 0.0, "Gaussian raw selected rho")
    with np.load(g.local(paper, GAUSSIAN_ROOT + "/gen_00000000/ensemble.npz"), allow_pickle=False) as blob:
        g.require(len(blob.files) == 2 and set(blob.files) == {"ensemble", "ids"}, "Gaussian NPZ fields differ")
        full = blob["ensemble"]
        g.require(full.shape == (480, 39, 6, 48) and full.dtype == np.float64
                  and np.isfinite(full).all(), "Gaussian payload shape/dtype/finite differs")
        g.require(blob["ids"].dtype.kind == "U" and blob["ids"].shape == (480,), "Gaussian ID dtype/shape differs")
        g.agree(blob["ids"].tolist(), list(ids), "Gaussian full ID order")
    decoded = full[list(INDICES), :, :5, 18:30].transpose(0, 1, 3, 2) * nrm.scale[:5] + nrm.mean[:5]
    g.require(np.array_equal(data["samples"][3], decoded), "Gaussian excerpt decode differs")
    del full

    methods, energies = {}, {}
    for m, method in enumerate(METHODS):
        neural = method in STEPS
        if neural:
            step = STEPS[method]
            source = f"artifacts/test/{method}_s7-g{step}/ensemble"
            chosen = comparison["selected"][method + "_s7"]["raw"]
            g.agree(chosen["generation"], step, method + " raw-selected step")
            for filename in ("COMMIT.json", "ensemble.npz"):
                key = source + "/gen_00000000/" + filename
                inputs[key] = real["input_sha256"][key]
            lineage = {"checkpoint_sha256": chosen["checkpoint_sha256"], "seed": 7,
                       "step": step, "n_eval": 50, "selection": "raw"}
            g.agree(scores["row_ids"][method + "_raw"], ["7", "13", "42"], "raw neural seed order")
            published = g.deadline_numeric(scores["data"][method + "_raw"])
            g.require(published.shape == (3, 480), "raw neural score shape differs")
            published = published[0]
        else:
            source = GAUSSIAN_ROOT
            lineage = {"control_fit_sha256": fit_hash, "fit_select_sha256": fit_select,
                       "seed": 13, "step": None, "control": control, "selection": "raw"}
            candidate = gaussian["candidates"]["lambda_0.1_rho_0"]
            g.agree(candidate["ensemble_commit_sha256"], inputs[source + "/gen_00000000/COMMIT.json"], "Gaussian score COMMIT")
            published = g.deadline_numeric([row["primary_es"] for row in candidate["raw"]["per_window"]])
            g.require(published.shape == (480,), "Gaussian score shape differs")
        methods[method] = {**lineage, "protocol_sha256": g.DEADLINE_PROTOCOL if neural else GAUSSIAN_PROTOCOL,
                           "n_generated": 5 if method == "G5" else 6, "n_draws": 39, "arrays": identity,
                           "ensemble_npz_sha256": inputs[source + "/gen_00000000/ensemble.npz"],
                           "ensemble_commit_sha256": inputs[source + "/gen_00000000/COMMIT.json"], "source": source}
        energies[method] = []
        for slot, index in enumerate(INDICES):
            value = _energy(data["samples"][m, slot], data["truth"][slot, 18:30, :5], nrm.mean, nrm.scale)
            g.require(math.isfinite(value) and abs(value - published[index]) <= 1e-12,
                      method + ": recomputed excerpt primary energy differs from published raw score")
            energies[method].append({"case_id": ids[index], "zero_based_index": index,
                                     "recomputed": value, "published": float(published[index])})
    expected = {"schema_version": 1, "analysis": "frozen_raw_imputation_examples", "selection": SELECTION,
                "case_ids": [ids[i] for i in INDICES], "method_order": list(METHODS), "methods": methods,
                "channels": CHANNELS, "units": UNITS, "arrays": identity, "input_sha256": inputs,
                "source_sha256": sources, "per_case_primary_energy": energies, "energy_absolute_tolerance": 1e-12,
                "energy_coordinates": "training-standardized hidden primary channels; all 39 raw draws",
                "truth_source": "physical observations, not stored ensemble flanks",
                "files": {"sample-excerpt.npz": g.digest(g.local(paper, excerpt_name))}}
    g.require(set(record) == set(expected), "excerpt manifest fields differ")
    _exact(g, {k: v for k, v in record.items() if k != "per_case_primary_energy"},
           {k: v for k, v in expected.items() if k != "per_case_primary_energy"}, "excerpt provenance")
    g.agree(record["per_case_primary_energy"], energies, "excerpt primary energy records")
    for method in METHODS:
        for row, actual in zip(record["per_case_primary_energy"][method], energies[method]):
            for key in ("recomputed", "published"):
                g.require(abs(row[key] - actual[key]) <= 1e-12, "excerpt energy absolute tolerance differs")

    fixed_name = "evidence/deadline/fixed-final.json"
    fixed = g.read_json(g.local(paper, fixed_name))
    rows = [{"coefficient": r["c_r"], "seed": r["seed"],
             "energy": r["results"]["raw"]["summary"]["es5"]["month_balanced"],
             "residual": r["results"]["raw"]["summary"]["r_obs_mean"]["month_balanced"]}
            for r in fixed["rows"]]
    coefficients = [0., .00005, .0005, .005]
    g.agree([(r["coefficient"], r["seed"]) for r in rows],
            [(c, s) for c in coefficients for s in (7, 13, 42)], "fixed-final plotted grid")
    means = [{"coefficient": c, **{key: math.fsum(r[key] / 3 for r in rows if r["coefficient"] == c)
                                  for key in ("energy", "residual")}} for c in coefficients]
    fixed_figure = _figure(g, paper, "figures/fixed_final", "fig_fixed_final",
        {"fixed_final": {"path": fixed_name, "sha256": g.digest(g.local(paper, fixed_name))}},
        {"plotted_rows": rows, "mean_rows": means, "scope": FIXED_SCOPE,
         "axes": {"x": {"scale": "categorical", "positions": [0, 1, 2, 3],
                        "coefficient_labels": ["0", "0.00005", "0.0005", "0.005"], "coefficients": coefficients},
                  "a": {"y_scale": "linear", "metric": "results.raw.summary.es5.month_balanced", "units": "TRAIN-standardized"},
                  "b": {"y_scale": "linear", "metric": "results.raw.summary.r_obs_mean.month_balanced",
                        "units": "nT^2", "quantity": "mean absolute observation residual"}}})
    for slot, index in enumerate(INDICES):
        directory = f"figures/imputation/case_{index}"
        raw = g.read_json(g.local(paper, directory + "/fig_imputation.manifest.json"))
        limits = g.deadline_numeric(raw["y_limits"])
        g.require(limits.shape == (5, 2) and np.all(limits[:, 0] < limits[:, 1]), "signed y limits invalid")
        coords = _coordinates(data["truth"][slot], data["samples"][:, slot])
        plotted = [data["truth"][slot, :, :5]]
        for values in coords["methods"].values():
            plotted.extend(np.asarray(values[k]).reshape(-1, 5) for k in ("low", "median", "high", "draws"))
        plotted = np.concatenate(plotted)
        g.require(np.all(limits[:, 0] <= plotted.min(axis=0)) and np.all(limits[:, 1] >= plotted.max(axis=0)),
                  "signed y limits clip plotted truth, bands or draws")
        _figure(g, paper, directory, "fig_imputation",
            {"excerpt": {"path": excerpt_name, "sha256": g.digest(g.local(paper, excerpt_name))},
             "provenance": {"path": manifest_name, "sha256": g.digest(g.local(paper, manifest_name))}},
            {"case_index": index, "case_id": ids[index], "channels": CHANNELS, "units": UNITS,
             "method_order": list(METHODS), "selection": SELECTION, "coordinates": coords,
             "scope": IMPUTATION_SCOPE, "y_limits": limits.tolist(),
             "y_scale": [{"scale": "linear"}] * 4 + [
                 {"scale": "symlog", "linthresh": 0.05, "ticks": [-1.0, -0.1, 0.0, 0.1, 1.0]}]})
    detail_cases = []
    for slot, index in enumerate(INDICES):
        coords = _coordinates(data["truth"][slot], data["samples"][:, slot])
        ordered = np.sort(data["samples"][:, slot, :, :, 4], axis=1)
        limits = {}
        for start, stop in ((0, 3), (3, 4)):
            values = np.concatenate((data["truth"][slot, 16:32, 4],
                                     ordered[start:stop, (1, 37), :].reshape(-1)))
            lo, hi = float(values.min()), float(values.max())
            pad = 0.08 * max(hi - lo, 1e-6)
            for name in METHODS[start:stop]:
                limits[name] = [lo - pad, hi + pad]
        detail_cases.append({"case_index": index, "coordinates": coords,
                             "x_limits": [-10, 65], "y_limits": limits})
        target = data["truth"][slot, 18:30, 4]
        q = np.sort(data["samples"][:, slot, :, :, 4], axis=1)
        covered = ((q[:, 1] <= target) & (target <= q[:, 37])).sum(axis=1).tolist()
        g.require(covered == ([0, 0, 0, 12] if index == 120 else [0, 1, 2, 12]),
                  "local variance coverage counts differ")
        g.require(np.all((q[:3, 19] - target).mean(axis=1) > 0)
                  and np.all(((q[:3, 1] <= target) & (target <= q[:3, 37])).mean(axis=1) < .9),
                  "local neural variance overprediction/undercoverage claim differs")
        g.require(np.all((q[3, 1] <= target) & (target <= q[3, 37]))
                  and np.any(q[3, 1] < 0)
                  and np.all((q[3, 37] - q[3, 1]).mean() > (q[:3, 37] - q[:3, 1]).mean(axis=1)),
                  "local Gaussian coverage/width/support claim differs")
    _figure(g, paper, "figures/imputation/detail", "fig_imputation",
            {"excerpt": {"path": excerpt_name, "sha256": g.digest(g.local(paper, excerpt_name))},
             "provenance": {"path": manifest_name, "sha256": g.digest(g.local(paper, manifest_name))}},
            {"cases": detail_cases, "y_scale": "linear",
             "limit_rule": "truth bins 16:32 and ranks 1,37; neural methods share limits, Gaussian separate; padding 8% of range, floor 1e-6",
             "axis_sides": {"neural": "left", "gaussian": "right"}})
    return {"excerpt_sha256": expected["files"]["sample-excerpt.npz"],
            "case_indices": list(INDICES), "case_ids": expected["case_ids"],
            "methods": list(METHODS), "primary_energies": {k: [r["recomputed"] for r in v] for k, v in energies.items()},
            "fixed_final_rows": len(rows), "fixed_final_means": fixed_figure["mean_rows"],
            "figure_manifests": {k: PINS[k] for k in PINS if k.startswith("figures/")},
            "scope": "Pinned audited excerpt; eight energies and four figure coordinates verified. "
                     "Public Gaussian excerpt decoded; neural lineage hash-bound, not full neural replay."}
