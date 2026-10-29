"""Bounded synthetic final-gate contracts, never today's manuscript as a fixture.

No denoising, training, network, private artifacts or TeX installation is needed.
The integration fixture uses a real, minimal PDF read by pdftotext/pdfinfo.
"""

import copy
import hashlib
import json
import re
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pikepdf
import pytest

from experiments import verify_prigm_paper as gate
from prigm.artifacts import id_hash
from prigm.recovery import ContractMismatch

ERRORS = (ValueError, KeyError, ContractMismatch)


def save(path, record):
    path.write_text(json.dumps(record, allow_nan=False))
    return gate.digest(path)


def fake_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


REVISED_SOURCES = (
    "prigm/artifacts.py", "prigm/run_cycle.py", "prigm/run_deep_diagnostics.py",
    "prigm/run_persistence.py", "prigm/run_score.py", "prigm/run_secondaries.py",
    "prigm/run_strata.py", "prigm/scoring.py",
)


@pytest.fixture
def source_revision_copy(tmp_path, monkeypatch):
    archive = Path("reproducibility/accepted_source")
    shutil.copytree(gate.REPO / archive, tmp_path / archive)
    documentation = Path(gate.documentation.ARCHIVE)
    shutil.copytree(gate.REPO / documentation, tmp_path / documentation)
    later_documentation = Path(gate.documentation.OCT29_ARCHIVE)
    shutil.copytree(gate.REPO / later_documentation, tmp_path / later_documentation)
    (tmp_path / "prigm").mkdir()
    for name in (*REVISED_SOURCES, "prigm/config.py"):
        shutil.copyfile(gate.REPO / name, tmp_path / name)
    monkeypatch.setattr(gate, "REPO", tmp_path)
    return tmp_path, tmp_path / archive


@pytest.mark.parametrize("name", REVISED_SOURCES)
@pytest.mark.parametrize("damage", [None, "current", "snapshot", "missing_snapshot", "old_current"])
def test_execution_source_requires_both_reviewed_files(source_revision_copy, name, damage):
    root, archive = source_revision_copy
    original = (archive / name).read_bytes()
    assert (root / name).read_bytes() != original
    if damage in ("current", "snapshot"):
        path = (root if damage == "current" else archive) / name
        path.write_bytes(path.read_bytes() + b"\n# unreviewed change\n")
    elif damage == "missing_snapshot":
        (archive / name).unlink()
    elif damage == "old_current":
        (root / name).write_bytes(original)
    if damage is None:
        assert gate.execution_source_hash(name) == hashlib.sha256(original).hexdigest()
    else:
        with pytest.raises((*ERRORS, OSError)):
            gate.execution_source_hash(name)


def test_execution_source_refuses_self_consistent_manifest_rewrite(source_revision_copy):
    root, archive = source_revision_copy
    name = "prigm/scoring.py"
    manifest = gate.read_json(archive / "MANIFEST.json")
    for parent, field in ((root, "accepted_sha256"), (archive, "execution_sha256")):
        path = parent / name
        path.write_bytes(path.read_bytes() + b"\n# forged revision\n")
        manifest["files"][name][field] = gate.digest(path)
    save(archive / "MANIFEST.json", manifest)
    with pytest.raises(ERRORS, match="accepted source manifest"):
        gate.execution_source_hash(name)



def test_execution_source_parses_the_manifest_bytes_it_authenticated(source_revision_copy, monkeypatch):
    root, archive = source_revision_copy
    name = "prigm/artifacts.py"
    manifest = archive / "MANIFEST.json"
    original = manifest.read_bytes()
    replacement = json.loads(original)
    current = root / name
    current.write_bytes(current.read_bytes() + b"\n# unreviewed revision\n")
    replacement["files"][name]["accepted_sha256"] = gate.digest(current)
    replacement_bytes = json.dumps(replacement).encode()
    original_read = Path.read_bytes
    calls = []

    def swap_on_second_read(path):
        if path == manifest:
            calls.append(path)
            return original if len(calls) == 1 else replacement_bytes
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", swap_on_second_read)
    with pytest.raises(ERRORS, match="reviewed current source|source provenance mismatch"):
        gate.execution_source_hash(name)
    assert len(calls) == 1


def test_unregistered_source_keeps_exact_current_identity(source_revision_copy):
    root, _ = source_revision_copy
    name = "prigm/config.py"
    original = gate.digest(root / name)
    assert gate.execution_source_hash(name) == original
    (root / name).write_bytes((root / name).read_bytes() + b"\n# unregistered change\n")
    with pytest.raises(ERRORS):
        gate.agree(original, gate.execution_source_hash(name))


def test_frozen_reference_refuses_current_hash_as_historical(tmp_path):
    oracle = {"source_hashes": {"scoring.py": gate.digest(gate.REPO / "prigm/scoring.py")}}
    with pytest.raises(ERRORS, match="frozen oracle source"):
        gate.replay_frozen_toy_reference(tmp_path / "unused.npz", oracle, fake_hash("oracle"))


def write_pdf(path, pages, author="", uri=None, subject=""):
    """Small valid ASCII PDF with searchable text, using only the standard library."""
    objects = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        page_id, content_id = len(objects) + 1, len(objects) + 2
        kids.append(f"{page_id} 0 R")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 900 1000] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode())
        escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
        content = ("BT /F1 10 Tf 30 960 Td 14 TL\n" + "\n".join(f"({line}) Tj T*" for line in escaped) + "\nET").encode("ascii")
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    if uri is not None:
        annotation_id = len(objects) + 1
        objects[3] = objects[3][:-2] + f" /Annots [{annotation_id} 0 R] >>".encode()
        escaped_uri = uri.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        objects.append((f"<< /Type /Annot /Subtype /Link /Rect [30 900 850 920] "
                        f"/A << /S /URI /URI ({escaped_uri}) >> >>").encode("ascii"))
    objects.append(f"<< /Author ({author}) /Title (Synthetic final gate) /Subject ({subject}) >>".encode())
    result, offsets = b"%PDF-1.4\n", [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(result)
    result += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    result += f"trailer << /Size {len(objects) + 1} /Root 1 0 R /Info {len(objects)} 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    path.write_bytes(result)


def table(label, rows, caption):
    return ("\\begin{table}\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
            + "\\begin{tabular}{lrrrrrr}\nProcedure & Values \\\\\n"
            + "\n".join((" & ".join(row) if isinstance(row, tuple) else row) + " & " + " & ".join(values) + r" \\" for row, values in rows.items())
            + "\n\\end{tabular}\n\\end{table}\n")


def original_manifest(root, filename, source, protocol_hash):
    sha = gate.digest(root / filename)
    save(root / "MANIFEST.json", {"schema_version": 1, "source_protocol_sha256": protocol_hash,
         "scope": "Synthetic fixed population.", "transformation": "Byte-identical.",
         "validation": "Fixture identity, not a scientific execution.",
         "files": {filename: sha}, "records": {filename: {"source_record": source, "source_sha256": sha}}})


def sensitivity_fixture(root, protocol, summary, matrices, observations, inputs):
    directory = root / "evidence/sensitivity"
    directory.mkdir()
    z, ids, nrm, identity = observations
    features = gate.sensitivity.physical_features(z, nrm.b_ref_sq)
    # Invented TRAIN features; these are not fitted to the TEST population.
    ranges = {"context_cancellation": (.55, .83), "target_absolute_cancellation": (.35, .75),
              "context_energy": (.7, 1.4), "target_energy": (.7, 1.4)}
    train = {k: np.linspace(*ranges.get(k, (.1, 2.)), protocol["arrays"]["TRAIN"]["n_windows"])
             for k in gate.sensitivity.FEATURES}
    cutoffs = gate.sensitivity.fit_cutoffs(train)
    methods = {}
    for label, diag in summary["methods"].items():
        methods[label] = {k: diag[k] for k in ("procedure", "scoring_kind", "selection_kind", "generations", "fixed_model_seeds", "n_realizations")}
        methods[label]["files"] = {seed: {"path": f["primary"], "sha256": inputs[f["primary"]],
            "checkpoint_sha256": diag["checkpoint_sha256"][seed], "control_fit_sha256": f["control_fit_sha256"],
            "ensemble_commit_sha256": f["ensemble_commit_sha256"]} for seed, f in diag["files"].items()}
    from experiments import run_prigm_toy_repair as toy
    inference = {name: gate.execution_source_hash(name) for name in toy.source_identity()}
    producers = {**inference, **{name: gate.execution_source_hash(name) for name in
                  ("analysis/prigm_training_strata.py", "analysis/prigm_followup_summary.py")}}
    record = {"schema_version": 1, "analysis_source_sha256": producers["analysis/prigm_training_strata.py"],
        "producer_source_sha256": producers, "inference_source_sha256": inference,
        **{k: summary[k] for k in ("protocol_sha256", "selection_sha256", "test_started_sha256", "ids", "months")},
        "ids_sha256": id_hash(ids), "input_sha256": inputs,
        "arrays": {k: protocol["arrays"][k] for k in ("TRAIN", "TEST")},
        "normalization": {"mean": nrm.mean.tolist(), "scale": nrm.scale.tolist(), "b_ref_sq": nrm.b_ref_sq,
                          "n_bins": nrm.n_bins, "sha256": identity["normalization_sha256"]},
        "inventory_counts": summary["input_manifest"]["inventory_counts"],
        "recorded_raw_sha256_by_year": summary["input_manifest"]["recorded_raw_sha256_by_year"],
        "methods": methods, "cutoffs": cutoffs, "features_in_test_id_order": {k: v.tolist() for k, v in features.items()},
        "cutoff_definitions": {"target_indices": list(range(18, 30)), "context_indices": list(range(18)) + list(range(30, 48)),
            "energy": "Window mean of Q=(m^2+v_s)/B_ref_sq over the stated bins.",
            "dimensionless_cancellation": "Window mean of C=(m^2-sum(mu^2))/m^2; bin ratio is zero when m^2=0.",
            "test_edge_tie_counts": {k: [int(np.count_nonzero(features[k] == e)) for e in v["values"]] for k, v in cutoffs.items()}},
        "scope": gate.sensitivity.SCOPE, "bootstrap": summary["bootstrap"],
        **gate.sensitivity.summarize_groups(matrices, np.asarray(summary["months"]), ids, features, cutoffs)}
    save(directory / "training-strata.json", record)
    original_manifest(directory, "training-strata.json", "artifacts/training-strata-01.json", summary["protocol_sha256"])


def learned_fixture(root, training, oracle):
    from analysis import export_prigm_toy_evidence as export
    from experiments import run_prigm_toy_repair as toy
    directory = root / "evidence/toy/learned"
    directory.mkdir()
    identity = gate.load_arrays(root / "evidence/toy/toy.npz", "TOY")[3]
    inputs = {k: fake_hash(k) for k in ("PROTOCOL.json", "PROTOCOL_RECEIPT.json", "TRAINED.json", "EVALUATED.json")}
    inputs["artifacts/toy.npz"] = identity["sha256"]
    originals = {key: inputs[path] for key, path in (("protocol", "PROTOCOL.json"), ("trained", "TRAINED.json"),
                                                   ("evaluated", "EVALUATED.json"), ("array", "artifacts/toy.npz"))}
    originals["oracle"] = gate.digest(root / "evidence/toy/oracle-stationary-01.json")
    execution = {name: gate.execution_source_hash(name) for name in toy.source_identity()}
    sources = {"training": execution, "inference": execution, "coordinator_sha256": gate.documentation.source_hash(gate.REPO, "experiments/run_prigm_toy_repair.py")}
    recipe = toy.fixed_recipe()
    settings = toy.scientific_settings(fake_hash("snapshot"))
    protocol = {"study": "controlled_stationary_toy_repair", "exposure": "historical_test_exposed", "boundary_law": "stationary",
        "recipe": recipe, "window_law": "stationary_boundary_markov_bridge_with_repeated_endpoint_flanks; not a full 48-bin Markov trajectory",
        "samplers": {"TOY": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}}, "sampling_chunk_windows": 8,
        "state_convention": toy.STATE_CONVENTION, "noise_convention": toy.NOISE_CONVENTION,
        "selection": "fixed final checkpoint at update 5000 only; no validation or coefficient selection; all nine fits retained"}
    fits = {}
    for index, fit in enumerate(recipe["fits"]):
        name = fit["name"]
        run, parent = f"runs/{name}", f"artifacts/toy/{name}-g5000"
        config = {k: v for k, v in recipe.items() if k not in ("fits", "milestones")}
        config.update({k: v for k, v in fit.items() if k != "name"})
        config.update(contract_version=2, source_hashes=sources["training"], data_hashes=training["data_hashes"],
                      normalization_hash=training["normalization_sha256"], settings=settings)
        row = {"config": config, "generation_updates": 5000}
        for key, path in (("checkpoint_sha256", f"{run}/gen_00005000/state.pt"),
                          ("checkpoint_commit_sha256", f"{run}/gen_00005000/COMMIT.json"),
                          ("ensemble_commit_sha256", f"{parent}/ensemble/gen_00000000/COMMIT.json"),
                          ("completed_sha256", f"{parent}/COMPLETED.json")):
            row[key] = inputs[path] = fake_hash(path)
        for path in (f"{run}/gen_00005000/meta.json", f"{parent}/REQUEST.json", f"{parent}/ensemble/gen_00000000/ensemble.npz",
                     f"artifacts/training/{name}-request.json", f"artifacts/training/{name}-completed.json"):
            inputs[path] = fake_hash(path)
        inputs[f"{run}/source_snapshot.zip"] = settings["source_snapshot_sha256"]
        for offset in range(0, 288, 8):
            for file in ("COMMIT.json", "ensemble.npz"):
                path = f"{parent}/chunk-{offset:04d}/gen_00000000/{file}"
                inputs[path] = fake_hash(path)
        for corrected in (False, True):
            kind = "corrected" if corrected else "raw"
            records = []
            for i in range(288):
                value = .11 + index * .03 + i / 10000 + corrected * .001
                record = {k: value for k in export.SCALAR_METRICS}
                record.update({k: np.full((12, 5), value).tolist() for k in export.ARRAY_METRICS})
                record["variogram"] = -.02  # An unbiased finite-ensemble score may be negative.
                records.append(record)
            scores = [r["es5"] for r in records]
            reduced = {k: {w: np.mean([r[k] for r in records], axis=0).tolist() for w in ("mean", "month_balanced")}
                       for k in records[0]}
            variant = {"corrected": corrected, "per_window": records, "secondary": reduced, "scores": scores,
                       "mean_energy": float(np.mean(scores)), "month_balanced_energy": float(np.mean(scores)), "source_sha256": {}}
            for key in ("score", "secondaries"):
                path = f"{parent}/{key}-{kind}.json"
                variant["source_sha256"][key] = inputs[path] = fake_hash(path)
            row[kind] = variant
        fits[name] = row
    learned = {"schema_version": 1, "original_sha256": originals, "sources": sources, "arrays": identity,
               "protocol": protocol, "training_identity": training, "training_settings": settings,
               "n": 288, "n_draws": 39, "ids": [f"TOY:{i}" for i in range(288)], "months": [0] * 288,
               "interval_protocol": export.INTERVAL_PROTOCOL, "fit_order": list(fits), "fits": fits}
    before_replay = copy.deepcopy(oracle)
    reference = gate.replay_frozen_toy_reference(root / "evidence/toy/toy.npz", oracle, originals["oracle"])
    assert oracle == before_replay
    reference["source_protocol_sha256"] = originals["protocol"]
    outputs = {name: save(directory / name, record) for name, record in
               (("learned.json", learned), ("reference-calibration.json", reference))}
    save(directory / "MANIFEST.json", {"schema_version": 1, "exporter_source_sha256": gate.documentation.source_hash(gate.REPO, "analysis/export_prigm_toy_evidence.py"),
         "sources": sources, "original_sha256": originals, "input_sha256": inputs, "output_sha256": outputs})


def gradient_fixture(protocol, inputs):
    """Invented geometry with real field names; never a production gradient claim."""
    gradients = {"schema_version": 1, "protocol_sha256": protocol["source_protocol_sha256"],
                 "scope": gate.exporter.GRADIENT_SCOPE, "grids": {}}
    source_hashes = {"prigm/run_gradient_probe.py": fake_hash("probe source")}
    schedule = {"steps": 100, "array_hashes": {"a": fake_hash("schedule")}}
    inputs["run_probe_grid.py"] = fake_hash("probe launcher")
    for generation in (2868, 29636):
        parent = f"artifacts/gradient-g{generation}"
        request = {"protocol_sha256": gradients["protocol_sha256"], "generation": generation,
                   "arm": "O-S", "seeds": [7, 13, 42], "splits": ["TRAIN", "VAL"],
                   "n_windows_per_split": 8, "noise_seed": 7003, "c_r": .0005,
                   "launcher_sha256": inputs["run_probe_grid.py"], "checkpoint_sha256": {}}
        grid = {"request": request, "reports": {}}
        for kind, filename in (("request", "REQUEST.json"), ("completion", "COMPLETED.json")):
            inputs[f"{parent}/{filename}"] = grid[f"source_{kind}_sha256"] = fake_hash((parent, filename))
        for seed in (7, 13, 42):
            checkpoint_sha = fake_hash(("O-S", seed, "raw") if generation == 2868 else (generation, seed))
            request["checkpoint_sha256"][str(seed)] = checkpoint_sha
            inputs[f"runs/O-S_s{seed}/gen_{generation:08d}/state.pt"] = checkpoint_sha
            for split in ("TRAIN", "VAL"):
                filename = f"O-S_s{seed}-{split.lower()}.json"
                inputs[f"{parent}/{filename}"] = fake_hash((parent, filename, "original bytes"))
                identity = protocol["arrays"][split]
                selected_ids = [f"{split}:{i}" for i in range(8)]
                grid["reports"][filename] = {
                    "schema_version": 1, "diagnostic": "weighted_physics_data_parameter_gradients",
                    "generation_updates": generation, "checkpoint_sha256": checkpoint_sha, "c_r": .0005,
                    "arrays": copy.deepcopy(identity), "arrays_path": f"artifacts/inventory/{split.lower()}.npz",
                    "run_root": f"runs/O-S_s{seed}", "current_source_hashes": dict(source_hashes),
                    "training_source_hashes": dict(source_hashes),
                    "training_contract": {"arm": "O-S", "seed": seed, "c_r": .0005, "contract_version": 2,
                        "estimator": "S", "residual": "obs", "n_generated": 6, "batch_size": 128,
                        "max_updates": 29636, "widths": [64, 128, 128],
                        "n_examples": protocol["arrays"]["TRAIN"]["n_windows"],
                        "data_hashes": protocol["training_data_hashes"],
                        "normalization_hash": identity["normalization_sha256"], "source_hashes": dict(source_hashes),
                        "settings": {"diffusion": copy.deepcopy(schedule)}},
                    "normalization": {"sha256": identity["normalization_sha256"], "mean": [0.] * 6,
                                      "scale": [1.] * 6, "b_ref_sq": 1., "n_bins": 400},
                    "model": {"class": "prigm.model.ConditionalUNet", "widths": [64, 128, 128],
                              "n_generated": 6, "n_parameters": 1603398, "mode": "eval"},
                    "selection": {"n_requested": 8, "n_selected": 8,
                                  "rule": "np.linspace(0, n_available - 1, min(n_available, n_windows), dtype=int64); first index if count is one",
                                  "indices": np.linspace(0, identity["n_windows"] - 1, 8, dtype=np.int64).tolist(),
                                  "ids": selected_ids, "ids_sha256": id_hash(selected_ids)},
                    "noise": {"seed": 7003, "shape": [8, 6, 48], "device": "cpu", "dtype": "torch.float32",
                              "generator": "torch.Generator", "sha256": fake_hash("noise")},
                    "inputs": {"u0_sha256": fake_hash("u0"), "observed_sha256": fake_hash("observed"),
                               "hidden_slice": [18, 30], "state_convention": "clean_observed_flanks_hidden_diffusion"},
                    "schedule": {**copy.deepcopy(schedule), "probe_timesteps": [1, 25, 50, 100],
                                 "probe_values": {k: [.5] * 4 for k in ("a", "s", "w", "sigma")}},
                    "records": [{"t": t, "estimator": e, "residual": r, "weighted_data_mean": .5,
                                 "weighted_physics_mean": .25, "total_mean": .75, "data_gradient_norm": 2.,
                                 "physics_gradient_norm": 1., "physics_to_data_norm_ratio": .5,
                                 "gradient_cosine": 0., "gradient_angle_degrees": 90.}
                                for t in (1, 25, 50, 100) for e in ("M", "S") for r in ("inst", "obs")],
                }
        for report in grid["reports"].values():
            report["schedule"]["probe_values"].update(sigma=[.002, .02, .05, .9], w=[3., 4., .8, 4e-7])
            for row in report["records"]:
                factor = (1 + report["training_contract"]["seed"] / 20) * (1.7 if report["arrays"]["split"] == "VAL" else 1)
                factor *= (3 if generation == 2868 else 1) * (7 if row["residual"] == "inst" else 1) * (5 if row["estimator"] == "M" else 1)
                row["physics_gradient_norm"] = {1: .15, 25: .013, 50: .006, 100: 2.73e-5}[row["t"]] * factor
                row["data_gradient_norm"] = 1.17e-7 if row["t"] == 100 else 2.
                row["physics_to_data_norm_ratio"] = row["physics_gradient_norm"] / row["data_gradient_norm"]
        gradients["grids"][str(generation)] = grid
    return gradients


def followup_prose(learned, reference, gradients, strata):
    """Independent prose/table fixture from invented evidence, not production values."""
    rows = {}
    for name in learned["fit_order"]:
        fit = learned["fits"][name]
        metric = fit["raw"]["secondary"]
        rows[(fit["config"]["arm"], f"{fit['config']['c_r']:g}")] = [
            f"{fit['raw']['mean_energy']:.4f}", f"{fit['corrected']['mean_energy']:.4f}",
            f"{metric['r_obs_mean']['mean']:.4f}", f"{metric['frac_support_violation_beyond_rounding']['mean'] * 100:.1f}",
            f"{metric['cov90_mean']['mean'] * 100:.1f}"]
    ref = reference["secondary"]
    caption = ("Nine diffusion fits to the stationary hidden Markov model after 5{,}000 optimizer updates at seed 7. "
               "Evaluation uses 288 windows and 39 samples per window; clipping retains the model state and samples. Raw $|R|$ is the mean absolute residual of the observation identity in HMM units. "
               "The inequality column gives the percentage of generated bins that fail a moment inequality beyond rounding tolerance, separately from the equality residual. "
               "Coverage averages the five primary coordinates and twelve positions. "
               "The independent exact conditional ensemble has "
               f"{ref['cov90_mean']['mean'] * 100:g}\\% coverage for the inclusive nominal 90\\% intervals.")
    toy_prose = "Table compares nine fits to the stationary chain with 50 DDIM evaluations for sampling.\n\n" + table("tab:learnedtoy", rows, caption)
    toy_prose += (f"\nDiscreteness also matters for interval coverage. The exact conditional ensembles in Table 2 give coverage of {ref['cov90_mean']['mean'] * 100:.1f}\\% and "
                  f"{ref['cov50_mean']['mean'] * 100:.2f}\\% under the nominal 90\\% and 50\\% conventions, "
                  f"with mean widths {ref['width90_mean']['mean']:.4f} and {ref['width50_mean']['mean']:.4f} in standardized coordinates. "
                  "The intervals include their endpoints, so tied values can raise coverage above the values expected for a continuous distribution. This explains why the nominal percentages alone would mischaracterize calibration in this toy."
                  "")
    final = gradients["grids"]["29636"]
    values = final["reports"]["O-S_s7-train.json"]["schedule"]["probe_values"]
    coefficients = [.0005 / (2 * s * w) for s, w in zip(values["sigma"], values["w"])]
    def sci(v):
        m, e = f"{v:.3e}".split("e")
        return m + r"\times10^{" + str(int(e)) + "}"
    def span(split, t, key, scientific=False):
        numbers = [r[key] for s in (7, 13, 42) for r in final["reports"][f"O-S_s{s}-{split}.json"]["records"]
                   if r["t"] == t and r["residual"] == "obs" and r["estimator"] == "S"]
        pair = (min(numbers), max(numbers))
        return " to ".join(sci(v) for v in pair) if scientific else " to ".join(format(v, ".0f" if t == 100 else ".4f") for v in pair)
    ratio = "physics_to_data_norm_ratio"
    weight = ("To examine how the penalty enters optimization, the gradient comparison evaluates the weighted data and residual terms on the same batches before clipping the total gradient. "
              "For each seed, the same eight TRAIN and eight VAL windows are used at the first and final O-S checkpoints, with noise seed 7003 and $c_R=0.0005$. "
              "Both residuals and both estimators are evaluated at t in (1,25,50,100) without changing the model weights. "
              "A constant $c_R$ still gives different relative penalty weights across noise levels. "
              r"With $\lambda_t=c_R/(2\Sigma_t)$ and $c_R=0.0005$, the ratios $\lambda_t/w_t$ at $t=1,25,50,100$ are "
              + f"{coefficients[0]:.4f}, {coefficients[1]:.6f}, {coefficients[2]:.6f}, and {coefficients[3]:.1f}. "
              + "The large final ratio accompanies a Min-SNR data weight of $" + sci(values["w"][-1]) + "$. "
              + "To understand the effect on optimization, the derivatives of both loss terms also matter.\n\n"
              + "At the final O-S checkpoints, the ratio of residual to data gradient norms ranges from "
              + span("train", 100, ratio) + " on TRAIN and " + span("val", 100, ratio) + " on VAL at $t=100$. "
              + "At $t=25$, the ranges are " + span("train", 25, ratio) + " and " + span("val", 25, ratio) + ". "
              + "Each range spans three seeds, with eight windows per seed. "
              + "In absolute terms, however, the TRAIN residual-gradient norm at $t=100$ is only " + span("train", 100, "physics_gradient_norm", True)
              + ", compared with " + span("train", 1, "physics_gradient_norm") + " at $t=1$. "
              + "Thus, a large ratio can accompany a small gradient. "
              + "Optimizer updates combine noise levels and clip the total gradient afterward. These ratios describe the loss gradients at individual noise levels; their effect on prediction quality also depends on the rest of the training trajectory.")
    cutoffs, groups = strata["cutoffs"], strata["groups"]
    ch, cr, th, tr = [groups[k] for k in ("context_high", "context_rest", "target_high", "target_rest")]
    def vector(values, precision):
        return ",".join(format(v, f".{precision}f") for v in values)
    def word(n):
        return {5: "five", 7: "seven", 20: "twenty"}.get(n, str(n))
    pair = ch["comparisons"]["O-S_raw_minus_D_raw"]
    g5 = ch["comparisons"]["O-S_raw_minus_G5_raw"]
    target = th["comparisons"]["O-S_raw_minus_D_raw"]
    sparse = groups["target_q1_high"]
    cancellation = ("The groups use mean $C$ over the 36 available bins or mean $A$ over the twelve hidden bins. "
        + "High cancellation means strictly above the TRAIN 75th percentile: "
        + f"{cutoffs['context_cancellation']['values'][0]:.6f} for the available intervals and {cutoffs['target_absolute_cancellation']['values'][0]:.6f} for the missing interval. "
        + f"Every one of the {strata['arrays']['TRAIN']['n_windows']} overlapping training windows contributes equally to these thresholds, whereas normalization uses distinct bins. "
        + "The rules predate the implementation, which followed inspection of the primary results; the analysis is exploratory.\n\n"
        + "TRAIN quartile thresholds are (" + vector(cutoffs["context_energy"]["values"], 6) + ") for the available intervals and ("
        + vector(cutoffs["target_energy"]["values"], 6) + ") for the missing interval. "
        + "A value equal to an energy threshold enters the higher energy group. A value equal to the cancellation threshold enters the rest group. "
        + f"The {len(groups)} groups comprise the full test set, high cancellation and the rest, and combinations of cancellation with energy conditions. All {len(strata['methods'])} procedures are evaluated in each group. "
        + "Scores give equal weight to the months present and then average the three diffusion seeds.\n\n"
        + f"High cancellation in the available measurements identifies {ch['n_windows']} windows across all {word(ch['n_years'])} years. "
        + f"O-S minus D is {pair['mean_diff']:.4f}, with a 95\\% interval of [{vector(pair['year_95'], 4)}] from resampling years. "
        + "All three seed differences are negative, while the interval "
        + ("includes" if pair['year_95'][0] <= 0 <= pair['year_95'][1] else "excludes") + " zero. "
        + f"O-S minus G5 is {g5['mean_diff']:.4f}, with interval [{vector(g5['year_95'], 4)}], "
        + f"and O-S minus D is {cr['comparisons']['O-S_raw_minus_D_raw']['mean_diff']:.4f} in the remaining {cr['n_windows']} windows. "
        + "G5 has a lower mean score in this group, while the O-S versus D interval "
        + ("spans" if pair['year_95'][0] <= 0 <= pair['year_95'][1] else "excludes") + " zero. "
        + "The group uses information available to the model; cancellation can still vary with other magnetic conditions.\n\n"
        + f"High cancellation during the missing hour identifies {th['n_windows']} windows. O-S minus D is {target['mean_diff']:.4f}, with interval [{vector(target['year_95'], 4)}], "
        + f"compared with {tr['comparisons']['O-S_raw_minus_D_raw']['mean_diff']:.4f} in the remaining {tr['n_windows']} windows. "
        + "Although the thresholds come from TRAIN, assignment to these TEST groups uses the withheld values. The score comparison is consequently conditional on outcome selection. "
        + "Combining high cancellation during the missing hour with the lowest energy quartile leaves "
        + f"{word(sparse['n_windows'])} windows from {word(sparse['n_years'])} years. "
        + "Their scores describe this small subset of TEST outcomes. Interpreting them as evidence of conditional predictive superiority depends on accounting for the rule that uses withheld values to form the groups.")
    return rows, caption, {"Training diffusion models on the same HMM": toy_prose,
                           "How the residual penalty is weighted": weight, "Directional cancellation with thresholds from TRAIN": cancellation}


def fixture_visible(text):
    # ASCII PDF writer uses the spelled-out Greek names supported by prose_text.
    return gate.clean_cell(text.replace(r"\lambda", "lambda").replace(r"\Sigma", "Sigma")).replace("^", "")


def addition_lines(title, prose, number):
    def wrap(text):
        return textwrap.wrap(fixture_visible(text), width=100, break_long_words=False, break_on_hyphens=False)
    lines = [f"H.{number}  {title}"]
    if "tab:learnedtoy" in prose:
        block = gate.float_block(prose, "tab:learnedtoy")
        before, after = prose.split(block)
        lines += wrap(before)
        lines += wrap("Table 12: " + gate.float_caption(prose, "tab:learnedtoy"))
        lines += [""]
        lines += ["  ".join(row) for row in gate.source_table_rows(prose, "tab:learnedtoy") if row[0] != "Procedure"]
        lines += [""] + wrap(after)
    else:
        lines += wrap(prose)
    return lines


@pytest.fixture(scope="module")
def bundle_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("prigm-paper-fixture") / "paper"
    real, toy, data, figure = [root / suffix for suffix in ("evidence/real", "evidence/toy", "evidence/data", "figures/hmm")]
    for directory in (real, toy, data, figure):
        directory.mkdir(parents=True)
    shutil.copyfile(gate.PAPER / "neurips_2025.sty", root / "neurips_2025.sty")
    years = list(gate.bootstrap.YEARS)
    ids = [f"{y}-{m:02d}-{y}-{m:02d}-{d:02d}T00:00-s{52 if y <= 2014 else 71}" for y in years for m in (8, 9) for d in range(1, 13)]
    months = np.array([int(w[:4]) * 100 + int(w[5:7]) for w in ids])
    rng = np.random.default_rng(51)
    matrices = {f"{a}_{k}": 1 + rng.random((3, 480)) for a in gate.bootstrap.ARMS
                for k in ("raw", "corrected", "corrected_at_raw_checkpoint")}
    matrices.update({f"{a}_{k}": 1 + rng.random((1, 480)) for a in gate.bootstrap.CONTROLS for k in ("raw", "corrected")})
    # Ensure the manuscript's fixed-seed sign claim holds in this invented study.
    matrices["O-S_raw"] = matrices["D_raw"] - .075 + rng.normal(0, .003, (3, 480))
    identities = {split: {"sha256": fake_hash(split), "split": split, "n_windows": n,
                         "ids_sha256": id_hash(ids) if split == "TEST" else fake_hash(split + "ids"),
                         "normalization_sha256": fake_hash("normalization")}
                  for split, n in (("TRAIN", 200), ("VAL", 319), ("TEST", 480))}
    from prigm.observation import summaries_from_vectors
    observation_root = root / "evidence/observations"
    observation_root.mkdir()
    physical = summaries_from_vectors(rng.normal(size=(480, 48, 5, 3)) * 4)
    np.savez(observation_root / "test.npz", z=physical, ids=np.asarray(ids), split="TEST",
             mean=np.arange(6.) + .13, scale=np.arange(6.) + .7, b_ref_sq=48., n_bins=400)
    observations = gate.load_arrays(observation_root / "test.npz", "TEST")
    identities["TEST"] = observations[3]
    for identity in identities.values():
        identity["normalization_sha256"] = observations[3]["normalization_sha256"]
    protocol_hash, selection_hash, phase_hash = [fake_hash(n) for n in ("protocol", "selection", "phase")]
    original_manifest(observation_root, "test.npz", "artifacts/inventory/test.npz", protocol_hash)
    input_hashes = {name: fake_hash(name) for name in ("PIPELINE_COMPLETED.json", "CPU_FOLLOWUP_COMPLETED.json")}
    input_hashes.update({"PROTOCOL.json": protocol_hash, "SELECTION.json": selection_hash, "TEST_STARTED.json": phase_hash})
    for split, identity in identities.items():
        input_hashes[f"artifacts/inventory/{split.lower()}.npz"] = identity["sha256"]
    selected = {f"{a}_s{s}": {k: {"generation": 2868 if k == "raw" else 9560,
                 "checkpoint_sha256": fake_hash((a, s, k)), "score": 1.}
                 for k in ("raw", "corrected")} for a in gate.bootstrap.ARMS for s in (7, 13, 42)}
    methods, diagnostic_methods, row_ids = {}, {}, {}
    for label, values in matrices.items():
        arm, kind = label.split("_", 1)
        neural = arm in gate.bootstrap.ARMS
        keys = ["7", "13", "42"] if neural else ["shared"]
        scoring_kind = "raw" if kind == "raw" else "corrected"
        selection_kind = "raw" if kind == "corrected_at_raw_checkpoint" else kind
        meta = {"procedure": "diffusion" if neural else arm, "scoring_kind": scoring_kind,
                "fixed_model_seeds": [7, 13, 42] if neural else [], "score_files": {}}
        diag = {"procedure": meta["procedure"], "scoring_kind": scoring_kind, "fixed_model_seeds": meta["fixed_model_seeds"],
                "n_realizations": len(keys), "selection_kind": selection_kind if neural else None,
                "generations": {}, "checkpoint_sha256": {}, "files": {}, "seedwise": {}, "metrics": {}}
        if neural:
            meta.update(selection_kind=selection_kind, generations={})
        else:
            meta["n_baseline_realizations"] = 1
        for i, key in enumerate(keys):
            primary, secondary = [f"artifacts/{label}/{key}/{name}.json" for name in ("score", "secondaries")]
            meta["score_files"][key] = primary
            diag["files"][key] = {"primary": primary, "secondaries": secondary,
                                    "ensemble_commit_sha256": fake_hash(label + key), "control_fit_sha256": None}
            input_hashes.update({primary: fake_hash(primary), secondary: fake_hash(secondary)})
            diag["checkpoint_sha256"][key] = selected[f"{arm}_s{key}"][selection_kind]["checkpoint_sha256"] if neural else None
            if neural:
                meta["generations"][key] = diag["generations"][key] = selected[f"{arm}_s{key}"][selection_kind]["generation"]
            per_seed = {metric: {"mean": .1 + i * .01, "month_balanced": .2 + i * .01} for metric in gate.diagnostics.SCALARS}
            per_seed.update({metric: {weight: np.full((12, 5), .2 + i * .01).tolist()
                                     for weight in ("mean", "month_balanced")} for metric in gate.diagnostics.POSITION_CHANNEL})
            per_seed["es5"] = {k: v.tolist() for k, v in gate.diagnostics.reductions(values[i], months).items()}
            diag["seedwise"][key] = per_seed
        diag["metrics"] = {metric: {w: np.mean([diag["seedwise"][k][metric][w] for k in keys], axis=0).tolist()
                                     for w in ("mean", "month_balanced")} for metric in diag["seedwise"][keys[0]]}
        methods[label], diagnostic_methods[label], row_ids[label] = meta, diag, keys
    counts = {"train_starts": 200, "train_windows_saved": 200, "train_bin_union": 400, "val_selected": 319, "test_selected": 480}
    raw_hashes = {str(y): fake_hash(y) for y in years}
    summary = {"schema_version": 1, "analysis_source_sha256": gate.documentation.source_hash(gate.REPO, "analysis/prigm_followup_summary.py"),
               "protocol_sha256": protocol_hash, "selection_sha256": selection_hash, "test_started_sha256": phase_hash,
               "ids": ids, "months": months.tolist(), "arrays": identities["TEST"], "input_sha256": dict(input_hashes),
               "input_manifest": {"array_identities": identities, "inventory_counts": counts, "recorded_raw_sha256_by_year": raw_hashes},
               "methods": diagnostic_methods, "interval_protocol": gate.diagnostics.INTERVAL_PROTOCOL}
    summary.update(gate.diagnostics.energy_strata(matrices, months, ids, physical))
    summary["bootstrap"] = {"seed": gate.bootstrap.BOOTSTRAP_SEED,
                            "replicates": gate.bootstrap.BOOTSTRAP_REPLICATES, "generator": "PCG64"}
    comparison = {"schema_version": 2, "protocol_sha256": protocol_hash, "selection_sha256": selection_hash,
                  "test_started_sha256": phase_hash, "ids": ids, "months": months.tolist(), "arrays": identities["TEST"],
                  "methods": methods, "selected": selected, "source_comparison_sha256": fake_hash("original comparison"),
                  "exposure": "historical_test_exposed", "n_replicates": gate.bootstrap.BOOTSTRAP_REPLICATES,
                  "seed": gate.bootstrap.BOOTSTRAP_SEED, "fixed_training_seeds": [7, 13, 42], "multiplicity": "with_replacement_year_blocks",
                  "year_blocks": years, "two_year_blocks": gate.bootstrap.two_year_blocks(),
                  "month_counts": {str(y * 100 + m): 12 for y in years for m in (8, 9)},
                  **gate.exporter.COMPARISON_TEXT, **gate.bootstrap.compare(matrices, months)}
    scores = {"schema_version": 1, "protocol_sha256": protocol_hash, "ids": ids, "months": months.tolist(),
              "fixed_training_seeds": [7, 13, 42], "data": {k: v.tolist() for k, v in matrices.items()}, "row_ids": row_ids}
    protocol = {"schema_version": 1, "source_protocol_sha256": protocol_hash,
                "arms": list(gate.bootstrap.ARMS), "seeds": [7, 13, 42], "milestones": gate.MILESTONES,
                "exposure": "historical_test_exposed", "study": "fixed_recipe_corrected_sampling_followup",
                "batch_size": 128, "budget_updates": 29636, "c_r": {"D": 0., "I-S": .0005, "O-S": .0005, "G5": 0.},
                "samplers": {"VAL": {"n_eval": 25, "n_draws": 39, "eval_seed": 7}, "TEST": {"n_eval": 50, "n_draws": 39, "eval_seed": 7}},
                "selection": "raw and support-corrected selected independently", "controls": {"gaussian": {"rho": 0.}},
                "arrays": identities, "training_data_hashes": {"arrays_sha256": identities["TRAIN"]["sha256"]}}
    payloads = {"comparison.json": comparison, "diagnostics.json": summary, "scores.json": scores,
                "protocol-summary.json": protocol, "gradients.json": gradient_fixture(protocol, input_hashes)}
    output_hashes = {name: save(real / name, record) for name, record in payloads.items()}
    input_hashes[gate.exporter.SUMMARY] = output_hashes["diagnostics.json"]
    input_hashes[gate.exporter.COMPARISON] = comparison["source_comparison_sha256"]
    save(real / "MANIFEST.json", {"schema_version": 1, "exporter_source_sha256": gate.documentation.source_hash(gate.REPO, "analysis/export_prigm_paper_evidence.py"),
         "analysis_source_sha256": summary["analysis_source_sha256"], "output_sha256": output_hashes, "input_sha256": input_hashes,
         "original_sha256": {n: input_hashes[n] for n in ("PROTOCOL.json", "PIPELINE_COMPLETED.json", "CPU_FOLLOWUP_COMPLETED.json", gate.exporter.SUMMARY, gate.exporter.COMPARISON)}})

    # Real stationary contexts, but invented scores. No exact denoiser is called.
    from prigm.toy_eval import build_eval_windows
    from prigm.oracle import infer_context, oracle_metadata
    from paper.workshops.prigm import fig_hmm
    from experiments import run_prigm_toy_repair as toy_runner
    from prigm.oracle import posterior_hidden
    from prigm.toy import sample_mixture_paths, state_summary
    from prigm.config import ORACLE_REF_SEED
    from prigm.scoring import energy_score
    training, toy_normalizer = toy_runner.training_identity()
    z, alphas, ps = build_eval_windows()
    np.savez(toy / "toy.npz", z=z, alpha=alphas, p=ps, ids=np.array([f"TOY:{i}" for i in range(288)]),
             split="TOY", boundary_law="stationary", mean=toy_normalizer.mean, scale=toy_normalizer.scale,
             b_ref_sq=toy_normalizer.b_ref_sq, n_bins=toy_normalizer.n_bins)
    with np.load(toy / "toy.npz", allow_pickle=False) as blob:
        base = oracle_metadata(toy / "toy.npz", "pooled", "stationary", blob)
        base["source_hashes"] = {name: gate.execution_source_hash("prigm/" + name)
                                 for name in base["source_hashes"]}
    oracle = {**copy.deepcopy(base), "n": 288, "n_draws": 39, "contexts": [], "methods": {}, "reference_seed": ORACLE_REF_SEED}
    reference_scores = []
    reference_rng = np.random.Generator(np.random.PCG64(ORACLE_REF_SEED))
    for i, window in enumerate(z):
        a_value, a, b = infer_context(window, toy_normalizer)
        oracle["contexts"].append({"window": i, "alpha": a_value, "generating_p": float(ps[i]), "a": a, "b": b})
        post = posterior_hidden(a_value, None, a, b, toy_normalizer, conditioning="pooled", boundary_law="stationary")
        paths = sample_mixture_paths(post, reference_rng, 39)
        states = np.array([state_summary(a_value, s) for s in (0, 1)])
        reference_scores.append(energy_score(toy_normalizer.transform(states[paths])[..., :5].reshape(39, -1),
                                            toy_normalizer.transform(window[18:30])[:, :5].reshape(-1)))
    for i, (method, budget) in enumerate((("exact_reference", 0), *fig_hmm.SAMPLERS)):
        values = (np.mean(reference_scores) + (1. if method == "two_step" else .05 * i)
                  + np.linspace(-.005, .005, 288)).tolist()
        if method == "exact_reference":
            values = reference_scores
        oracle["methods"][method] = {"denoiser_evaluations_per_draw": budget, "total_denoiser_evaluations": budget * 288 * 39,
                                      "mean_energy": float(np.mean(values)), "scores": values,
                                      "mean_energy_by_generating_p": {str(p): float(np.asarray(values)[ps == p].mean()) for p in (.1, .25, .5)}}
    hmm = {**copy.deepcopy(base), "n_context": 144, "n_context_per_regime": 4, "n_paths": 128,
           "timesteps": [1, 25, 50, 100], "context": [], "context_by_t": {}}
    for t in hmm["timesteps"]:
        summary_t = {"n": 36, "mean_es_vs_ref": 5.12345, "mean_reference_self_energy_mc": 2.35567, "mean_abs_r_star": 1.23456e-5}
        hmm["context_by_t"][str(t)] = summary_t
        for regime in range(9):
            for i in range(4):
                hmm["context"].append({**oracle["contexts"][regime * 32 + i], "regime": regime, "t": t,
                    "denoiser_evaluations_per_endpoint": 1 if t == 1 else 2, "mean_es_vs_ref": summary_t["mean_es_vs_ref"],
                    "reference_self_energy_mc": summary_t["mean_reference_self_energy_mc"], "mean_abs_r_star": summary_t["mean_abs_r_star"]})
    input_specs = {name: {"path": f"{name}-stationary-01.json", "sha256": save(toy / f"{name}-stationary-01.json", record),
                           "source_hashes": record["source_hashes"]} for name, record in (("hmm", hmm), ("oracle", oracle))}
    context, samplers, reference, shared, residuals = fig_hmm.validate_evidence(hmm, oracle)
    (figure / "fig_hmm.pdf").write_bytes(b"synthetic figure PDF bytes")
    (figure / "fig_hmm.png").write_bytes(b"synthetic figure PNG bytes")
    shutil.copyfile(gate.PAPER / "prigm.mplstyle", root / "prigm.mplstyle")
    save(figure / "fig_hmm.manifest.json", {"schema_version": 2, "figure": "fig_hmm", "input_schema_version": 2,
         "source": {"path": "fig_hmm.py", "sha256": gate.digest(gate.PAPER / "fig_hmm.py")}, "inputs": input_specs,
         "style": {"path": "prigm.mplstyle", "sha256": gate.digest(root / "prigm.mplstyle")},
         "toy_eval_sha256": gate.digest(toy / "toy.npz"), "boundary_law": "stationary", "conditioning": "pooled",
         "hmm_n_context": 144, "hmm_n_paths": 128, "verified_common_source_hashes": {k: base["source_hashes"][k] for k in shared},
         "plotted_rows": {"panel_a_context": context,
                          "panel_b_context_residuals": [{"t": row["t"], "n": row["n"],
                              "mean_abs_r_star": residuals[str(row["t"])]} for row in context],
                          "panel_c_samplers": samplers, "panel_c_horizontal_reference": reference},
         "output_sha256": {name: gate.digest(figure / name) for name in ("fig_hmm.pdf", "fig_hmm.png")}})

    quality = {"raw_inputs": {str(y): {"sha256": raw_hashes[str(y)], "path": f"omni_min{y}.asc", "n_minutes": 525600} for y in years},
               "train_starts_by_year": {str(y): 10 for y in years}, "candidate_train_grid_starts_by_year": {str(y): 100 for y in years},
               "eligible_train_starts": 200, "candidate_train_grid_starts": 2000, "eligible_start_frac_of_candidate_grid": .1,
               "eligible_bins_scanned": 400, "train_role_minutes": 1000, "train_valid_magnetic_minutes": 900,
               "dbot_known_minutes": 800, "dbot_negative_minutes": 700, "bins_with_any_negative_dbot": 300,
               "distinct_potential_target_bins": 250,
               "population_definitions": {"eligible_start_frac_of_candidate_grid":
                   "eligible_train_starts / candidate_train_grid_starts; overlapping starts, not independent trajectories."}}
    q15 = {"schema_version": 2, "raw_inputs": quality["raw_inputs"], "n_blocks": 40, "population_n_blocks": 40,
           "population_blocks_sha256": fake_hash("blocks"), "selection": {"method": "all", "max_blocks": None, "blocks_sha256": fake_hash("blocks")},
           "population_by_year": {str(y): {"eligible_blocks": 2, "eligible_train_starts": 10,
                "eligible_train_bins": 20, "source_block_counts": {"52": 2}} for y in years},
           "population_source_block_counts": {"52": 40},
           "max_abs_err_vs_raw": 2.351e-13, "mean_only_loss_mean_nT2": .123456}
    q15["selection"]["by_year"] = {str(y): {"n_blocks": 2, "source_block_counts": {"52": 2}} for y in years}
    inventory = {**counts, "hashes": raw_hashes}
    data_records = {"quality.json": quality, "q15.json": q15, "inventory.json": inventory}
    public_hashes = {name: save(data / name, record) for name, record in data_records.items()}
    save(data / "MANIFEST.json", {"schema_version": 1, **gate.DATA_DECLARATIONS,
         "files": public_hashes, "records": {name: {"source_record": original,
             "source_sha256": public_hashes[name] if name == "inventory.json" else fake_hash(original)}
             for name, original in gate.DATA_RECORDS.items()}})
    main_rows = {}
    for row, label in gate.MAIN_ROWS.items():
        raw, corrected = label + "_raw", label + "_corrected"
        m = diagnostic_methods[raw]["metrics"]
        main_rows[row] = [f"{comparison['arm_means'][raw]:.3f}", f"{comparison['arm_means'][corrected]:.3f}",
                          f"{100 * m['cov90_mean']['month_balanced']:.1f}", f"{100 * diagnostic_methods[corrected]['metrics']['cov90_mean']['month_balanced']:.1f}",
                          f"{m['r_obs_mean']['month_balanced']:.3f}", f"{100 * m['frac_support_violation_beyond_rounding']['month_balanced']:.1f}"]
    means = {k: v["mean_energy"] for k, v in oracle["methods"].items()}
    oracle_rows = {row: [f"{means[k]:.4f}"] for row, k in gate.ORACLE_ROWS.items()}
    paragraph = ("An exact denoiser isolates error in the sampling approximation. "
                 + r"Across 36 replicate contexts at $t=100$, two evaluations give mean absolute $R_{\mathrm{obs}}=1.23\times10^{-5}$ in HMM units, yet energy score 5.123 versus "
                 + "2.356 for independent exact conditional samples. "
                 + "On a separate set of 288 windows, two evaluations again give a poor score, while 25, 50, or 100 DDIM evaluations approach the exact conditional ensemble. "
                 + "These estimates have finite sampling error.")
    number_values = (200, 2000, 1000, 900, 800, 700, 300, 400, 250, 40, 10)
    count_text = " ".join(f"{v:,}".replace(",", "{,}") for v in number_values) + r" 10.00\% 2.35\times10^{-13} 0.1235"
    q15_text = (r"This property also holds numerically for all 40 eligible intervals of 15 minutes that begin on a UTC quarter hour within the union of training windows: the calculation from three bins differs from that using the original fifteen vectors by at most $2.35\times10^{-13}\,\mathrm{nT}^2$. "
                + r"Replacing each bin by its mean vector discards its vector variance and reduces the result by $0.1235\,\mathrm{nT}^2$ on average.")
    tex = (r"\documentclass{article}" + "\n" + r"\usepackage[dblblindworkshop]{neurips_2025}" + "\n"
           + r"\workshoptitle{2nd Workshop on Principles of Generative Modeling}" + "\n"
           + r"\makeatletter\renewcommand{\@noticestring}{" + gate.NOTICE + r"}\makeatother" + "\n"
           + r"\title{Synthetic final gate}\author{Anonymous}\begin{document}\maketitle" + "\n\n"
           + "Historical conclusions were withdrawn. " + r"\citep{fixture}" + "\n\n" + paragraph + "\n\n"
           + table("tab:main", main_rows, "Independently-selected correction is distinct from same-checkpoint repair. Inequality-only failures beyond rounding, not full realizability.")
           + r"\includegraphics{figures/hmm/fig_hmm.pdf}" + "\n"
           + r"\bibliography{references}\bibliographystyle{plainnat}\appendix\section{Appendix}" + "\n\n"
           + count_text + "\n\n" + q15_text + "\n\n" + table("tab:oracle", oracle_rows, "Stationary pooled reference.") + r"\end{document}")
    (root / "main.tex").write_text(tex)
    (root / "references.bib").write_text("@article{fixture,\n title={Synthetic citation title},\n author={Someone},\n year={2026}\n}\n")
    visible_paragraph = gate.clean_cell(paragraph).replace("^", "")
    # Keep lines narrow enough for extraction; source and PDF values are identical.
    pages = [["Synthetic final gate", gate.NOTICE, "Historical conclusions were withdrawn.", *textwrap.wrap(visible_paragraph, width=100, break_long_words=False, break_on_hyphens=False),
              *textwrap.wrap("Table 1: " + gate.clean_cell(gate.float_caption(tex, "tab:main")), width=100),
              *[row + "  " + "  ".join(values) for row, values in main_rows.items()]],
             ["Methods"], ["Results"], ["Discussion"], ["References", "[1] Someone. Synthetic citation title. 2026."],
             ["A  Appendix", " ".join(map(str, number_values)), "10.00% 2.35x10-13 0.1235", *textwrap.wrap(fixture_visible(q15_text), width=100), "Table 2: Stationary pooled reference.",
              *[row + "  " + "  ".join(values) for row, values in oracle_rows.items()]]]
    details = gate.detailed_table_values((comparison, summary), observations)
    for number, (label, rows) in enumerate(details.items(), 3):
        caption = "Synthetic " + label.split(":")[1] + " diagnostics."
        tex = tex.replace(r"\end{document}", table(label, rows, caption) + r"\end{document}")
        pages.append([f"Table {number}: {caption}", "Procedure  Values", *[
            "        " + ("  ".join(row) if isinstance(row, tuple) else row) + "  " + "  ".join(values)
            for row, values in rows.items()]])
    sensitivity_fixture(root, protocol, summary, matrices, observations, input_hashes)
    learned_fixture(root, training, oracle)
    learned = gate.read_json(toy / "learned/learned.json")
    reference = gate.read_json(toy / "learned/reference-calibration.json")
    strata = gate.read_json(root / "evidence/sensitivity/training-strata.json")
    learned_rows, learned_caption, additions = followup_prose(learned, reference, payloads["gradients.json"], strata)
    addition_pages = {}
    for i, (title, prose) in enumerate(additions.items(), 1):
        tex = tex.replace(r"\end{document}", r"\subsection{" + title + "}\n\n" + prose + "\n\n" + r"\end{document}")
        addition_pages[title] = len(pages)
        pages.append(addition_lines(title, prose, i))
    tex = tex.replace(r"\end{document}", r"\section{End diagnostics}\end{document}")
    pages.append(["Z  End diagnostics"])
    (root / "main.tex").write_text(tex)
    write_pdf(root / "main.pdf", pages)
    return {"root": root, "real": real, "toy": toy, "data": data, "figure": figure,
            "pages": pages, "main_rows": main_rows, "oracle_rows": oracle_rows, "details": details,
            "learned_rows": learned_rows, "additions": additions, "addition_pages": addition_pages}


@pytest.fixture
def bundle(bundle_template, tmp_path):
    root = tmp_path / "paper"
    shutil.copytree(bundle_template["root"], root)
    result = copy.deepcopy(bundle_template)
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = root / value.relative_to(bundle_template["root"])
    try:
        yield result
    finally:
        # Each case owns its copy. Keep disk use bounded even on failed assertions.
        shutil.rmtree(root, ignore_errors=True)


def mutate(path, function):
    record = json.loads(path.read_text())
    function(record)
    save(path, record)


def rebind_real(bundle, name):
    root = bundle["real"]
    def update(manifest):
        sha = gate.digest(root / name)
        manifest["output_sha256"][name] = sha
        if name == "diagnostics.json":
            manifest["original_sha256"][gate.exporter.SUMMARY] = sha
            manifest["input_sha256"][gate.exporter.SUMMARY] = sha
    mutate(root / "MANIFEST.json", update)


def rebind_data(bundle, name):
    def update(manifest):
        public = gate.digest(bundle["data"] / name)
        manifest["files"][name] = public
        if name == "inventory.json":
            manifest["records"][name]["source_sha256"] = public
    mutate(bundle["data"] / "MANIFEST.json", update)


def rebind_toy(bundle, name):
    def update(manifest):
        record = json.loads((bundle["toy"] / f"{name}-stationary-01.json").read_text())
        manifest["inputs"][name]["sha256"] = gate.digest(bundle["toy"] / f"{name}-stationary-01.json")
        manifest["inputs"][name]["source_hashes"] = record["source_hashes"]
    mutate(bundle["figure"] / "fig_hmm.manifest.json", update)


def snapshot(root):
    return {p.relative_to(root).as_posix(): gate.digest(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.skipif(not shutil.which("pdftotext") or not shutil.which("pdfinfo"), reason="PDF tools unavailable")
def test_complete_synthetic_cli_passes_read_only(bundle, capsys):
    before = snapshot(bundle["root"])
    assert gate.main(["--paper-dir", str(bundle["root"])]) == 0
    output = capsys.readouterr().out
    assert "OVERALL: PASS" in output and "494 cells / 534 scalars" in output
    assert "twelve tables" in output and "learned toy table: 45 cells" in output
    assert "relative residual weight: coefficients" in output and "cancellation groups: cutoffs" in output
    assert snapshot(bundle["root"]) == before


@pytest.mark.parametrize("damage", ["mean", "seed_mean", "year_ci", "two_year_ci", "relative_ci", "flag", "flag_type", "family", "exposure", "replicates", "selected_generation"])
def test_rebound_public_comparison_mutations_fail(bundle, damage):
    def change(row):
        pair = row["comparisons"]["O-S_raw_minus_D_raw"]
        if damage == "mean":
            row["arm_means"]["D_raw"] += .01
        elif damage == "seed_mean":
            row["seed_means"]["D_raw"]["7"] += .01
        elif damage in ("year_ci", "two_year_ci", "relative_ci"):
            pair[{"year_ci": "year_95", "two_year_ci": "twoyear_95", "relative_ci": "rel_year_95"}[damage]][0] += .01
        elif damage == "flag":
            pair["seeds_agree_neg"] = not pair["seeds_agree_neg"]
        elif damage == "flag_type":
            pair["seeds_agree_neg"] = int(pair["seeds_agree_neg"])
        elif damage == "family":
            pair["family"] = "blind_primary"
        elif damage == "exposure":
            row["exposure"] = "blind"
        elif damage == "replicates":
            row["n_replicates"] = 9999
        else:
            row["selected"]["D_s7"]["raw"]["generation"] = 9560
    mutate(bundle["real"] / "comparison.json", change)
    rebind_real(bundle, "comparison.json")
    with pytest.raises(ERRORS):
        gate.validate_real(bundle["root"])


@pytest.mark.parametrize("damage", ["seedwise", "metric", "interval", "missing_seed", "checkpoint", "selection", "private", "absolute_path"])
def test_rebound_diagnostic_mutations_fail(bundle, damage):
    def change(row):
        method = row["methods"]["D_raw"]
        if damage == "seedwise":
            method["seedwise"]["7"]["es5"]["mean"] += .01
        elif damage == "metric":
            method["metrics"]["cov90_mean"]["month_balanced"] += .01
        elif damage == "interval":
            row["interval_protocol"] = "pooled_117_draws"
        elif damage == "missing_seed":
            del method["seedwise"]["13"]
        elif damage == "checkpoint":
            method["checkpoint_sha256"]["7"] = fake_hash("wrong checkpoint")
        elif damage == "selection":
            method["selection_kind"] = "corrected"
        elif damage == "private":
            row["note"] = "Kevin Lee"
        else:
            row["note"] = "/tmp/private/result.json"
    mutate(bundle["real"] / "diagnostics.json", change)
    rebind_real(bundle, "diagnostics.json")
    with pytest.raises(ERRORS):
        gate.validate_real(bundle["root"])


@pytest.mark.parametrize("damage", ["ids", "duplicate_id", "missing_method", "controls", "seeds", "row_ids", "months", "nonfinite"])
def test_public_score_population_mutations_fail(bundle, damage):
    def change(row):
        if damage == "ids":
            row["ids"][0], row["ids"][1] = row["ids"][1], row["ids"][0]
        elif damage == "duplicate_id":
            row["ids"][1] = row["ids"][0]
        elif damage == "missing_method":
            del row["data"]["G5_corrected_at_raw_checkpoint"]
        elif damage == "controls":
            row["data"]["gaussian_raw"] *= 3
        elif damage == "seeds":
            row["fixed_training_seeds"] = [7, 13, 43]
        elif damage == "row_ids":
            row["row_ids"]["D_raw"] = ["13", "7", "42"]
        elif damage == "months":
            row["months"][0] = 200509
        else:
            row["data"]["D_raw"][0][0] = "NaN"
    mutate(bundle["real"] / "scores.json", change)
    rebind_real(bundle, "scores.json")
    with pytest.raises(ERRORS):
        gate.validate_real(bundle["root"])


@pytest.mark.parametrize("key,value", [("milestones", [2868]), ("c_r", {"D": 0., "G5": 0., "I-S": .005, "O-S": .005}),
                                      ("seeds", [7, 13, 41]), ("budget_updates", 29635), ("selection", "shared selected checkpoint")])
def test_protocol_recipe_is_fixed(bundle, key, value):
    mutate(bundle["real"] / "protocol-summary.json", lambda row: row.update({key: value}))
    rebind_real(bundle, "protocol-summary.json")
    with pytest.raises(ERRORS):
        gate.validate_real(bundle["root"])


@pytest.mark.parametrize("filename", sorted(gate.PAYLOADS))
def test_public_payload_hash_drift(bundle, filename):
    path = bundle["real"] / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ERRORS, match="hash"):
        gate.validate_real(bundle["root"])


@pytest.mark.parametrize("name", ["exporter_source_sha256", "analysis_source_sha256"])
def test_public_generator_hash_drift(bundle, name):
    mutate(bundle["real"] / "MANIFEST.json", lambda row: row.update({name: "0" * 64}))
    with pytest.raises(ERRORS):
        gate.validate_real(bundle["root"])


@pytest.mark.parametrize("damage", ["law", "conditioning", "n", "K", "scores", "mean", "source", "prior", "contexts"])
def test_stationary_oracle_adverse_mutations(bundle, damage):
    def change(row):
        if damage == "law":
            row["boundary_law"] = "iid"
        elif damage == "conditioning":
            row["conditioning"] = "regime-known"
        elif damage == "n":
            row["n"] = 287
        elif damage == "K":
            row["n_draws"] = 38
        elif damage == "scores":
            row["methods"]["ddim_50"]["scores"][0] += 1.
        elif damage == "mean":
            row["methods"]["exact_reference"]["mean_energy"] += 1.
        elif damage == "source":
            row["source_hashes"]["oracle.py"] = fake_hash("old source")
        elif damage == "prior":
            row["p_prior"]["0.1"] = .5
        else:
            row["contexts"][0]["a"] = 1 - row["contexts"][0]["a"]
    mutate(bundle["toy"] / "oracle-stationary-01.json", change)
    rebind_toy(bundle, "oracle")
    with pytest.raises(ERRORS):
        gate.validate_toy(bundle["root"])


@pytest.mark.parametrize("damage", ["paths", "count", "per_t", "mean", "duplicate", "regime"])
def test_hmm_population_and_reductions(bundle, damage):
    def change(row):
        if damage == "paths":
            row["n_paths"] = 127
        elif damage == "count":
            row["n_context"] = 143
        elif damage == "per_t":
            row["context_by_t"]["100"]["n"] = 35
        elif damage == "mean":
            row["context_by_t"]["100"]["mean_es_vs_ref"] += .1
        elif damage == "duplicate":
            row["context"][0] = row["context"][1]
        else:
            row["context"][0]["regime"] = 8
    mutate(bundle["toy"] / "hmm-stationary-01.json", change)
    rebind_toy(bundle, "hmm")
    with pytest.raises(ERRORS):
        gate.validate_toy(bundle["root"])


@pytest.mark.parametrize("damage", ["generator", "point", "residual_point", "output", "npz"])
def test_figure_and_npz_hash_bindings(bundle, damage):
    if damage == "npz":
        path = bundle["toy"] / "toy.npz"
        path.write_bytes(path.read_bytes() + b" ")
    elif damage == "output":
        (bundle["figure"] / "fig_hmm.pdf").write_bytes(b"stale historical figure")
    else:
        def change(row):
            if damage == "generator":
                row["source"]["sha256"] = fake_hash("unknown generator")
            elif damage == "residual_point":
                row["plotted_rows"]["panel_b_context_residuals"][0]["mean_abs_r_star"] += 1e-18
            else:
                row["plotted_rows"]["panel_c_samplers"][0]["mean_energy"] += 1e-14
        mutate(bundle["figure"] / "fig_hmm.manifest.json", change)
    with pytest.raises(ERRORS):
        gate.validate_toy(bundle["root"])


def test_supplied_python_is_never_executed(bundle):
    (bundle["root"] / "fig_hmm.py").write_text("raise AssertionError('untrusted code executed')\n")
    gate.validate_toy(bundle["root"])


@pytest.mark.parametrize("damage", ["cell", "duplicate", "missing", "unknown", "header", "comment", "caption"])
def test_main_table_cell_inventory_and_comment_decoys(bundle, damage):
    tex = (bundle["root"] / "main.tex").read_text()
    row = "D & " + " & ".join(bundle["main_rows"]["D"]) + r" \\"
    if damage == "cell":
        tex = tex.replace(row, row.replace(bundle["main_rows"]["D"][0], "9.999", 1))
    elif damage == "duplicate":
        tex = tex.replace(row, row + "\n" + row)
    elif damage == "missing":
        tex = tex.replace(row, "")
    elif damage == "unknown":
        tex = tex.replace(row, row + "\nUnknown & 9.999 \\\\")
    elif damage == "header":
        tex = tex.replace(row, row + "\n & 9.999 \\\\")
    elif damage == "comment":
        tex = "% " + row + "\n" + tex.replace(row, row.replace(bundle["main_rows"]["D"][0], "9.999", 1))
    else:
        tex = tex.replace("not full realizability", "full realizability")
    with pytest.raises(ERRORS):
        gate.main_table(tex, gate.pdf_checks(bundle["root"]), gate.validate_real(bundle["root"]))


@pytest.mark.parametrize("key", list(gate.ORACLE_ROWS.values()))
def test_oracle_table_rounding_bound_to_each_method(bundle, key):
    tex = (bundle["root"] / "main.tex").read_text()
    name = next(name for name, method in gate.ORACLE_ROWS.items() if method == key)
    tex = tex.replace(name + " & " + bundle["oracle_rows"][name][0], name + " & 9.9999")
    with pytest.raises(ERRORS):
        gate.toy_claims(tex, gate.pdf_checks(bundle["root"]), gate.validate_toy(bundle["root"]))


@pytest.mark.parametrize("old,new", [("energy score 5.123", "energy score 5.124"),
                                    ("versus 2.356", "versus 2.357"),
                                    (r"1.23\times10^{-5}", r"1.23\times10^{-6}"),
                                    ("two evaluations again give a poor score", "two evaluations again give an excellent score"),
                                    ("25, 50, or 100 DDIM evaluations", "25, 51, or 100 DDIM evaluations")])
def test_main_toy_prose_mutations_cannot_hide_in_valid_table(bundle, old, new):
    tex = (bundle["root"] / "main.tex").read_text()
    assert old in tex
    tex = tex.replace(old, new)
    with pytest.raises(ERRORS):
        gate.toy_claims(tex, gate.pdf_checks(bundle["root"]), gate.validate_toy(bundle["root"]))


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("anchor,old,new", [
    ("Across 36 replicate contexts", "energy score 5.123", "energy score 5.124"),
    ("Across 36 replicate contexts", "versus 2.356", "versus 2.357"),
    ("On a separate set", "again give a poor score", "again give an excellent score"),
])
def test_bad_toy_comparison_cannot_hide_behind_valid_duplicate(bundle_template, scope, anchor, old, new):
    root = bundle_template["root"]
    tex = (root / "main.tex").read_text()
    rendered = extracted(bundle_template["pages"])
    paragraph = next(p for p in tex.split("\n\n") if "An exact denoiser isolates error" in p)
    clause = paragraph[paragraph.index(anchor):].split(". ", 1)[0] + "."
    assert clause.count(old) == 1
    evidence = gate.validate_toy(root)
    gate.toy_claims(tex, rendered, evidence)
    if scope == "source":
        assert tex.count(clause) == 1
        tex = tex.replace(clause, clause.replace(old, new) + " " + clause, 1)
    else:
        rendered, count = re.subn(r"\s+".join(map(re.escape, old.split())), new, rendered)
        assert count == 1
        rendered = rendered.replace("References", gate.clean_cell(clause).replace("^", "") + "\nReferences", 1)
    with pytest.raises(ERRORS, match=scope + ": missing or duplicate clause"):
        gate.toy_claims(tex, rendered, evidence)


@pytest.mark.parametrize("marker", ["in progress", "NOT READY", "TODO", "TBD"])
def test_unresolved_draft_source_and_pdf_refused(bundle, marker):
    tex = (bundle["root"] / "main.tex").read_text()
    with pytest.raises(ERRORS, match="draft"):
        gate.source_checks(bundle["root"], tex + "\n" + marker)
    pages = copy.deepcopy(bundle["pages"])
    pages[0].append(marker)
    write_pdf(bundle["root"] / "main.pdf", pages)
    with pytest.raises(ERRORS, match="draft"):
        gate.pdf_checks(bundle["root"])


def test_commented_markers_do_not_count_and_withdrawn_disclosure_allowed(bundle):
    tex = gate.strip_comments((bundle["root"] / "main.tex").read_text() + "\n% TODO obsolete\n")
    gate.source_checks(bundle["root"], tex)


@pytest.fixture
def camera_ready_bundle(bundle):
    path = bundle["root"] / "main.tex"
    tex = (path.read_text().replace(gate.STYLE_ANONYMOUS, gate.STYLE_CAMERA_READY)
           .replace(gate.NOTICE_ANONYMOUS, gate.NOTICE_CAMERA_READY)
           .replace(r"\workshoptitle{2nd Workshop on Principles of Generative Modeling}",
                    r"\workshoptitle{2nd Workshop on Principles of Generative Modeling (PriGM)}")
           .replace(r"\author{Anonymous}", r"\usepackage{orcidlink}\hypersetup{pdfauthor={Kevin Lee}}"
                    r"\author{Kevin Lee\orcidlink{0009-0004-0388-9260}"
                    r"\\University of California at Los Angeles"
                    r"\\NASA Jet Propulsion Laboratory"
                    r"\\kevinlee69720@g.ucla.edu}"))
    path.write_text(tex)
    pages = copy.deepcopy(bundle["pages"])
    pages[0] = [line.replace(gate.NOTICE_ANONYMOUS, gate.NOTICE_CAMERA_READY) for line in pages[0]]
    pages[0].extend([gate.AUTHOR_CAMERA_READY, gate.EMAIL_CAMERA_READY, *gate.AFFILIATION_MARKERS])
    return tex, pages


@pytest.mark.parametrize("leak", [None, "other@example.edu", "other@example.com"])
def test_camera_ready_source_and_pdf_allow_only_declared_email(bundle, camera_ready_bundle, leak):
    tex, pages = camera_ready_bundle
    if leak is None:
        gate.source_checks(bundle["root"], tex)
        write_pdf(bundle["root"] / "main.pdf", pages, author=gate.AUTHOR_CAMERA_READY)
        gate.pdf_checks(bundle["root"])
    else:
        with pytest.raises(ValueError, match="email"):
            gate.source_checks(bundle["root"], tex + "\n" + leak)
        pages[0].append(leak)
        write_pdf(bundle["root"] / "main.pdf", pages, author=gate.AUTHOR_CAMERA_READY)
        with pytest.raises(ValueError, match="email"):
            gate.pdf_checks(bundle["root"])


@pytest.mark.parametrize("damage", ["style", "plot_style", "anonymous", "notice", "figure", "main_table", "appendix", "duplicate_label", "input"])
def test_source_contract_failures(bundle, damage):
    tex = (bundle["root"] / "main.tex").read_text()
    if damage == "style":
        (bundle["root"] / "neurips_2025.sty").write_text("changed vendor")
    elif damage == "plot_style":
        (bundle["root"] / "prigm.mplstyle").write_text("font.family: sans-serif\n")
    elif damage == "anonymous":
        tex = tex.replace(r"\author{Anonymous}", r"\author{Kevin Lee}")
    elif damage == "notice":
        tex = tex.replace("NeurIPS 2026", "NeurIPS 2025")
    elif damage == "figure":
        tex = tex.replace("figures/hmm/fig_hmm.pdf", "fig_hmm.pdf")
    elif damage == "main_table":
        tex = tex.replace(r"\label{tab:main}", "% \\label{tab:main}\n")
    elif damage == "appendix":
        tex = tex.replace(r"\appendix", "")
    elif damage == "duplicate_label":
        tex += r"\label{tab:oracle}"
    else:
        tex += r"\input{outside}"
    with pytest.raises(ERRORS):
        gate.source_checks(bundle["root"], gate.strip_comments(tex))


@pytest.mark.parametrize("damage", ["overlength", "references", "appendix", "author", "private", "notice", "draft"])
def test_real_pdf_text_metadata_and_pagination(bundle, damage):
    pages = copy.deepcopy(bundle["pages"])
    author = ""
    if damage == "overlength":
        pages.insert(4, ["Extra main text"])
    elif damage == "references":
        pages[4][0] = "A sentence mentioning References is not a heading."
    elif damage == "appendix":
        pages[0].insert(0, "A  Appendix")
    elif damage == "author":
        author = "A Private Person"
    elif damage == "private":
        pages[3].append("/home/private/file")
    elif damage == "notice":
        pages[0] = [line.replace("NeurIPS 2026", "NeurIPS 2025") for line in pages[0]]
    else:
        pages[1].append("TBD")
    write_pdf(bundle["root"] / "main.pdf", pages, author)
    with pytest.raises(ERRORS):
        gate.pdf_checks(bundle["root"])


@pytest.mark.parametrize("mode,case,valid", [
    ("anonymous", "no_ack", True), ("camera_ready", "no_ack", True),
    ("anonymous", "hidden", True), ("anonymous", "clean", False),
    *[("camera_ready", case, valid) for case, valid in (
        ("clean", True), ("fourth_page", True), ("next_reference_page", True), ("wrapped", True),
        ("orphan", False), ("missing_heading", False), ("renamed_heading", False),
        ("duplicate_heading", False), ("duplicate_source", False),
        ("unmatched_begin", False), ("unmatched_end", False), ("empty", False), ("macro", False),
        ("source_after_references", False), ("source_body_after_ack", False),
        ("source_mismatch", False), ("pdf_mismatch", False), ("missing_body", False),
        ("fifth_page_body", False), ("fifth_page_number", False), ("sixth_page", False),
        ("extra_pdf_body", False), ("extra_pdf_number", False), ("references_before_ack", False),
    )],
])
def test_pdf_acknowledgment_requires_matching_source_and_preserves_main_limit(bundle, request, mode, case, valid):
    if mode == "camera_ready":
        tex, pages = request.getfixturevalue("camera_ready_bundle")
    else:
        tex, pages = (bundle["root"] / "main.tex").read_text(), copy.deepcopy(bundle["pages"])
    heading = "Acknowledgments and Disclosure of Funding"
    body = ("Shared research infrastructure (ARES) supported this work under contract 12345. "
            "OMNI supplied the data. The author has no competing interests.")
    block = "\\begin{ack}\n" + body + "\n\\end{ack}\n"
    bibliography = r"\bibliography{references}"
    tex = tex.replace(bibliography, block + bibliography)
    ack_lines = [heading, *textwrap.wrap(body, width=85)]
    pages[4][:0] = ack_lines
    if case in ("no_ack", "orphan"):
        tex = tex.replace(block, "")
    if case in ("no_ack", "hidden"):
        del pages[4][:len(ack_lines)]
    elif case == "fourth_page":
        del pages[4][:len(ack_lines)]
        pages[3].extend(ack_lines)
    elif case == "next_reference_page":
        del pages[4][:len(ack_lines)]
        pages.insert(4, ack_lines + ["5"])
        tex = tex.replace(bibliography, r"\clearpage" + bibliography)
    elif case == "wrapped":
        continuation = "structure" + body.split("infrastructure", 1)[1]
        pages[4][:len(ack_lines)] = [heading, "Shared research infra-", *textwrap.wrap(continuation, width=85)]
    elif case == "missing_heading":
        pages[4].pop(0)
    elif case == "renamed_heading":
        pages[4][0] = "Acknowledgments"
    elif case == "duplicate_heading":
        pages[4].insert(0, heading)
    elif case == "duplicate_source":
        tex = tex.replace(block, block + block)
    elif case == "unmatched_begin":
        tex = tex.replace(r"\end{ack}", "")
    elif case == "unmatched_end":
        tex = tex.replace(r"\begin{ack}", "")
    elif case == "empty":
        tex = tex.replace(body, "")
    elif case == "macro":
        tex = tex.replace("(ARES)", r"(\textbf{ARES})")
    elif case == "source_after_references":
        tex = tex.replace(block + bibliography, bibliography + block)
    elif case == "source_body_after_ack":
        tex = tex.replace(bibliography, "Extra scientific results.\n" + bibliography)
    elif case == "source_mismatch":
        tex = tex.replace("contract 12345", "contract 54321")
    elif case == "pdf_mismatch":
        pages[4] = [line.replace("contract 12345", "contract 54321") for line in pages[4]]
    elif case == "missing_body":
        del pages[4][1:len(ack_lines)]
    elif case in ("fifth_page_body", "fifth_page_number"):
        pages[4].insert(0, "Extra scientific results." if case == "fifth_page_body" else "2026")
    elif case == "sixth_page":
        pages.insert(4, ["Extra scientific results."])
    elif case in ("extra_pdf_body", "extra_pdf_number"):
        pages[4].insert(len(ack_lines), "Extra scientific results." if case == "extra_pdf_body" else "2026")
    elif case == "references_before_ack":
        pages[4] = pages[4][len(ack_lines):] + ack_lines
    (bundle["root"] / "main.tex").write_text(tex)
    write_pdf(bundle["root"] / "main.pdf", pages, author=gate.AUTHOR_CAMERA_READY if mode == "camera_ready" else "")
    if valid:
        gate.pdf_checks(bundle["root"])
    else:
        with pytest.raises(ERRORS, match="acknowledgment|four pages"):
            gate.pdf_checks(bundle["root"])


@pytest.mark.parametrize("mode", ["anonymous", "camera_ready"])
@pytest.mark.parametrize("uri", [gate.PUBLIC_COMPANION_URL, "file:///home/private/study"])
def test_pdf_acknowledgment_does_not_exempt_uri_targets(bundle, request, mode, uri):
    if mode == "camera_ready":
        tex, pages = request.getfixturevalue("camera_ready_bundle")
    else:
        tex, pages = (bundle["root"] / "main.tex").read_text(), copy.deepcopy(bundle["pages"])
    body = "Shared facilities supported this work. No competing interests are declared."
    tex = tex.replace(r"\bibliography{references}", "\\begin{ack}\n" + body + "\n\\end{ack}\n"
                      + r"\bibliography{references}")
    if mode == "camera_ready":
        pages[4][:0] = ["Acknowledgments and Disclosure of Funding", body]
    (bundle["root"] / "main.tex").write_text(tex)
    write_pdf(bundle["root"] / "main.pdf", pages, author=gate.AUTHOR_CAMERA_READY if mode == "camera_ready" else "", uri=uri)
    if mode == "camera_ready" and uri == gate.PUBLIC_COMPANION_URL:
        gate.pdf_checks(bundle["root"])
    else:
        with pytest.raises(ERRORS, match="author-identifying|private"):
            gate.pdf_checks(bundle["root"])


@pytest.mark.skipif(not shutil.which("pdftotext") or not shutil.which("pdfinfo"), reason="PDF tools unavailable")
@pytest.mark.parametrize("mode", ["anonymous", "camera_ready"])
@pytest.mark.parametrize("case, valid", [
    ("clean", True), ("page_number", True), ("continued_references_new_page", True),
    ("missing", False), ("renamed", False), ("duplicated", False),
    ("before_reference", False), ("inline", False), ("missing_section", False),
    ("same_reference_page", False), ("continued_bibliography", False),
    ("body_before_heading", False), ("numeric_body_before_heading", False),
])
def test_pdf_appendix_heading_binds_first_source_section(tmp_path, mode, case, valid):
    camera_ready = mode == "camera_ready"
    style = gate.STYLE_CAMERA_READY if camera_ready else gate.STYLE_ANONYMOUS
    notice = gate.mode_notice(mode)
    author = gate.AUTHOR_CAMERA_READY if camera_ready else "Anonymous"
    workshop = "2nd Workshop on Principles of Generative Modeling" + (" (PriGM)" if camera_ready else "")
    tex = (r"\documentclass{article}" + "\n" + style + "\n"
           + r"\workshoptitle{" + workshop + "}\n"
           + r"\makeatletter\renewcommand{\@noticestring}{" + notice + r"}\makeatother" + "\n"
           + r"\title{Synthetic final gate}\author{" + author
           + r"}\begin{document}\maketitle\section{Main}\appendix" + "\n"
           + r"\section {Realizability and \textbf{resolved} energy}" + "\n"
           + r"\section{Later appendix}")
    heading = "A  Realizability   and resolved energy"
    pages = [["Synthetic final gate", notice,
              "A nonlinear support set need not contain a conditional mean.",
              "References are listed after the discussion."],
             ["Methods"], ["Results"], ["Discussion"],
             ["References", "[1] Synthetic reference."], [heading, "B  Later appendix"]]
    if camera_ready:
        pages[0].extend([author, gate.EMAIL_CAMERA_READY, *gate.AFFILIATION_MARKERS])
    if case == "page_number":
        pages[5].insert(0, "6")
    elif case == "continued_references_new_page":
        pages.insert(5, ["[2] Reference continued from the previous page."])
    elif case == "missing":
        pages[5].pop(0)
    elif case == "renamed":
        pages[5][0] = "A  Later appendix"
    elif case == "duplicated":
        pages[5].append(heading)
    elif case == "before_reference":
        pages[3].append(pages[5].pop(0))
    elif case == "inline":
        pages[5][0] += " is discussed here."
    elif case == "missing_section":
        tex = tex.split(r"\appendix")[0] + r"\appendix"
    elif case == "same_reference_page":
        pages[4].extend(pages.pop(5))
    elif case == "continued_bibliography":
        pages[5].insert(0, "[2] Reference continued from the previous page.")
    elif case == "body_before_heading":
        pages[5].insert(0, "Earlier body content on this page.")
    elif case == "numeric_body_before_heading":
        pages[5].insert(0, "2026")
    (tmp_path / "main.tex").write_text(tex)
    write_pdf(tmp_path / "main.pdf", pages, author=author if camera_ready else "")
    if valid:
        assert gate.pdf_checks(tmp_path)
    else:
        with pytest.raises(ERRORS, match="appendix"):
            gate.pdf_checks(tmp_path)


def test_pdf_row_mutation_not_rescued_by_original_values_elsewhere(bundle):
    pages = copy.deepcopy(bundle["pages"])
    row = next(i for i, line in enumerate(pages[0]) if line.startswith("D  "))
    old = pages[0][row]
    pages[0][row] = "D  9.999  9.999  99.9  99.9  9.999  99.9"
    pages[3].append("Numbers elsewhere " + old)
    write_pdf(bundle["root"] / "main.pdf", pages)
    with pytest.raises(ERRORS, match="PDF row"):
        gate.main_table((bundle["root"] / "main.tex").read_text(), gate.pdf_checks(bundle["root"]), gate.validate_real(bundle["root"]))


@pytest.mark.parametrize("damage", ["undefined", "uncited", "duplicate", "comment", "pdf_question", "pdf_title"])
def test_citations_bound_to_visible_source_and_rendered_bibliography(bundle, damage):
    tex = (bundle["root"] / "main.tex").read_text()
    rendered = gate.pdf_checks(bundle["root"])
    bib = bundle["root"] / "references.bib"
    if damage == "undefined":
        tex += r"\citep{missing}"
    elif damage == "uncited":
        bib.write_text(bib.read_text() + "@article{uncited, title={Never cited}}")
    elif damage == "duplicate":
        bib.write_text(bib.read_text() * 2)
    elif damage == "comment":
        tex = tex.replace(r"\citep{fixture}", "% \\citep{fixture}\n")
    elif damage == "pdf_question":
        rendered += "[?]"
    else:
        rendered = rendered.replace("Synthetic citation title", "Wrong rendered title")
    with pytest.raises(ERRORS):
        gate.citation_checks(bundle["root"], gate.strip_comments(tex), rendered)


def test_citation_apostrophe_typography_preserves_possessive(tmp_path):
    (tmp_path / "references.bib").write_text("@misc{fixture, title={Data at the {Earth}'s bow shock}}")
    tex = r"\citep{fixture}"
    rendered = "References\n[1] Data at the Earth\u2019s bow shock."
    gate.citation_checks(tmp_path, tex, rendered)
    with pytest.raises(ERRORS, match="cited title absent"):
        gate.citation_checks(tmp_path, tex, rendered.replace("\u2019", ""))


@pytest.mark.parametrize("name", ["main.pdf", "evidence/real/MANIFEST.json", "evidence/data/MANIFEST.json", "evidence/toy/oracle-stationary-01.json"])
def test_missing_mandatory_components_fail_default_cli(bundle, capsys, name):
    (bundle["root"] / name).unlink()
    before = snapshot(bundle["root"])
    assert gate.main(["--paper-dir", str(bundle["root"])]) == 1
    assert "OVERALL: FAIL" in capsys.readouterr().out
    assert snapshot(bundle["root"]) == before


def test_missing_paper_and_no_incomplete_switch(tmp_path, capsys):
    root = tmp_path / "nonexistent"
    assert gate.main(["--paper-dir", str(root)]) == 1
    assert "OVERALL: FAIL" in capsys.readouterr().out and not root.exists()
    with pytest.raises(SystemExit) as error:
        gate.main(["--allow-incomplete"])
    assert error.value.code == 2


def test_json_duplicate_and_nonfinite_refused(tmp_path):
    path = tmp_path / "bad.json"
    for payload in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        path.write_text(payload)
        with pytest.raises(ERRORS):
            gate.read_json(path)


@pytest.mark.parametrize("name,change", [
    ("quality.json", lambda r: r.update(eligible_train_starts=201)),
    ("quality.json", lambda r: r.update(eligible_start_frac_of_candidate_grid=.2)),
    ("quality.json", lambda r: r["raw_inputs"]["2005"].update(sha256="0" * 64)),
    ("quality.json", lambda r: r.update(note="/tmp/private/quality.json")),
    ("q15.json", lambda r: r.update(n_blocks=39)),
    ("q15.json", lambda r: r.update(max_abs_err_vs_raw=.1)),
    ("q15.json", lambda r: r["selection"].update(method="test_selected")),
    ("inventory.json", lambda r: r.update(train_bin_union=399)),
])
def test_anonymous_data_audit_mutations_fail(bundle, name, change):
    mutate(bundle["data"] / name, change)
    rebind_data(bundle, name)
    with pytest.raises(ERRORS):
        gate.data_audit(bundle["root"], (bundle["root"] / "main.tex").read_text(),
                        gate.pdf_checks(bundle["root"]), gate.validate_real(bundle["root"]))


def test_complete_hmm_regime_permutation_is_not_a_population_match(bundle):
    def change(row):
        for context in row["context"]:
            if context["regime"] in (0, 8):
                context["regime"] = 8 - context["regime"]
    mutate(bundle["toy"] / "hmm-stationary-01.json", change)
    rebind_toy(bundle, "hmm")
    with pytest.raises(ERRORS, match="regime"):
        gate.validate_toy(bundle["root"])


def test_pdf_toy_prose_error_cannot_hide_in_correct_oracle_table(bundle):
    pages = copy.deepcopy(bundle["pages"])
    pages[0] = [line.replace("energy score 5.123", "energy score 9.999") for line in pages[0]]
    write_pdf(bundle["root"] / "main.pdf", pages)
    with pytest.raises(ERRORS):
        gate.toy_claims((bundle["root"] / "main.tex").read_text(), gate.pdf_checks(bundle["root"]),
                        gate.validate_toy(bundle["root"]))


def test_main_table_must_also_render_before_references(bundle):
    pages = copy.deepcopy(bundle["pages"])
    rows = [line for line in pages[0] if any(line.startswith(label + "  ") for label in gate.MAIN_ROWS)]
    pages[0] = [line for line in pages[0] if line not in rows]
    pages[5].extend(rows)
    write_pdf(bundle["root"] / "main.pdf", pages)
    with pytest.raises(ERRORS):
        gate.main_table((bundle["root"] / "main.tex").read_text(), gate.pdf_checks(bundle["root"]),
                        gate.validate_real(bundle["root"]))


def test_pdf_caption_cannot_claim_full_realizability(bundle):
    pages = copy.deepcopy(bundle["pages"])
    pages[0] = [line.replace("not full realizability", "full realizability") for line in pages[0]]
    write_pdf(bundle["root"] / "main.pdf", pages)
    with pytest.raises(ERRORS, match="caption"):
        gate.main_table((bundle["root"] / "main.tex").read_text(), gate.pdf_checks(bundle["root"]),
                        gate.validate_real(bundle["root"]))


def test_title_mismatch_and_missing_pdf_tools_fail(bundle, monkeypatch):
    path = bundle["root"] / "main.tex"
    path.write_text(path.read_text().replace("Synthetic final gate", "A different paper title"))
    with pytest.raises(ERRORS, match="title"):
        gate.pdf_checks(bundle["root"])
    monkeypatch.setattr(gate.shutil, "which", lambda name: None)
    with pytest.raises(ERRORS, match="pdftotext"):
        gate.pdf_checks(bundle["root"])


def test_review_line_numbers_do_not_corrupt_pdf_bindings(bundle):
    pages = copy.deepcopy(bundle["pages"])
    line_number = 1
    for page in pages:
        for i, line in enumerate(page):
            page[i] = f"{line_number}            {line}"
            line_number += 1
    write_pdf(bundle["root"] / "main.pdf", pages)
    rendered = gate.pdf_checks(bundle["root"])
    tex = (bundle["root"] / "main.tex").read_text()
    gate.citation_checks(bundle["root"], tex, rendered)
    gate.main_table(tex, rendered, gate.validate_real(bundle["root"]))
    gate.toy_claims(tex, rendered, gate.validate_toy(bundle["root"]))


def test_external_symlink_refused(bundle, tmp_path):
    path = bundle["real"] / "scores.json"
    outside = tmp_path / "external.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ERRORS, match="escapes"):
        gate.validate_real(bundle["root"])


@pytest.mark.parametrize("damage", ["files", "records", "original_name", "source_hash", "public_hash",
                                    "inventory_changed", "reduced_equals_original", "scope", "transformation",
                                    "validation", "imaginary_source_pins", "extra_file"])
def test_declared_data_manifest_contract(bundle, damage):
    path = bundle["data"] / "MANIFEST.json"
    manifest = gate.read_json(path)
    if damage in ("files", "records"):
        del manifest[damage]["q15.json"]
    elif damage == "original_name":
        manifest["records"]["quality.json"]["source_record"] = "renamed-quality.json"
    elif damage == "source_hash":
        manifest["records"]["quality.json"]["source_sha256"] = "invalid"
    elif damage == "public_hash":
        manifest["files"]["q15.json"] = "invalid"
    elif damage == "inventory_changed":
        manifest["records"]["inventory.json"]["source_sha256"] = fake_hash("changed original")
    elif damage == "reduced_equals_original":
        manifest["records"]["quality.json"]["source_sha256"] = manifest["files"]["quality.json"]
    elif damage in gate.DATA_DECLARATIONS:
        manifest[damage] = "Raw files and original execution provenance independently reverified."
    elif damage == "imaginary_source_pins":
        manifest["source_sha256"] = {"prigm/quality.py": fake_hash("invented provenance")}
    else:
        save(bundle["data"] / "quality-01.json", {})
    save(path, manifest)
    with pytest.raises(ERRORS):
        gate.validate_data(bundle["root"])


@pytest.mark.parametrize("name", sorted(gate.DATA_RECORDS))
def test_data_public_byte_drift_is_not_an_original_record_check(bundle, name):
    path = bundle["data"] / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ERRORS, match="hash"):
        gate.validate_data(bundle["root"])


@pytest.mark.parametrize("name,change", [
    ("quality.json", lambda r: r["raw_inputs"]["2005"].update(path="data/omni_min2005.asc")),
    ("quality.json", lambda r: r["raw_inputs"]["2005"].update(n_minutes=0)),
    ("quality.json", lambda r: r["raw_inputs"].pop("2005")),
    ("q15.json", lambda r: r["population_by_year"]["2005"].update(eligible_train_starts=11)),
    ("q15.json", lambda r: r["population_by_year"]["2005"].update(eligible_train_bins=21)),
    ("q15.json", lambda r: r["selection"]["by_year"]["2005"].update(n_blocks=1)),
    ("q15.json", lambda r: r["population_source_block_counts"].update({"52": 41})),
])
def test_data_year_population_agreement(bundle, name, change):
    mutate(bundle["data"] / name, change)
    rebind_data(bundle, name)
    with pytest.raises(ERRORS):
        gate.data_audit(bundle["root"], (bundle["root"] / "main.tex").read_text(), "",
                        (None, gate.read_json(bundle["real"] / "diagnostics.json")))


def gradient_inputs(bundle):
    return (gate.read_json(bundle["real"] / "gradients.json"),
            gate.read_json(bundle["real"] / "MANIFEST.json")["input_sha256"],
            gate.read_json(bundle["real"] / "protocol-summary.json"))


@pytest.mark.parametrize("damage", ["grid", "report", "request_hash", "completion_hash", "source_report",
                                    "launcher", "checkpoint", "generation", "split", "noise_seed", "protocol",
                                    "source_map", "training_source", "array", "array_path", "private_run_root",
                                    "subset", "ids", "count", "duplicate_case", "missing_case"])
def test_public_gradient_internal_and_source_bindings(bundle, damage):
    gradients, inputs, protocol = gradient_inputs(bundle)
    grid = gradients["grids"]["29636"]
    report = grid["reports"]["O-S_s7-val.json"]
    if damage == "grid":
        del gradients["grids"]["2868"]
    elif damage == "report":
        del grid["reports"]["O-S_s42-train.json"]
    elif damage in ("request_hash", "completion_hash"):
        grid["source_" + damage.replace("_hash", "_sha256")] = fake_hash("wrong original")
    elif damage == "source_report":
        del inputs["artifacts/gradient-g29636/O-S_s7-val.json"]
    elif damage == "launcher":
        grid["request"]["launcher_sha256"] = fake_hash("wrong launcher")
    elif damage == "checkpoint":
        grid["request"]["checkpoint_sha256"]["7"] = fake_hash("wrong checkpoint")
    elif damage == "generation":
        report["generation_updates"] = 2868
    elif damage == "split":
        grid["request"]["splits"] = ["TRAIN", "TEST"]
    elif damage == "noise_seed":
        report["noise"]["seed"] = 7
    elif damage == "protocol":
        grid["request"]["protocol_sha256"] = fake_hash("wrong protocol")
    elif damage == "source_map":
        report["current_source_hashes"]["prigm/run_gradient_probe.py"] = fake_hash("different source")
    elif damage == "training_source":
        report["training_source_hashes"] = {"prigm/model.py": fake_hash("different training")}
    elif damage == "array":
        report["arrays"]["sha256"] = fake_hash("different arrays")
    elif damage == "array_path":
        report["arrays_path"] = "artifacts/inventory/test.npz"
    elif damage == "private_run_root":
        report["run_root"] = "/home/private/run"
    elif damage == "subset":
        report["selection"]["indices"][0] = 1
    elif damage == "ids":
        report["selection"]["ids"][0] = "VAL:other"
    elif damage == "count":
        report["selection"]["n_selected"] = 7
    elif damage == "duplicate_case":
        report["records"][0] = report["records"][1]
    else:
        report["records"].pop()
    with pytest.raises(ERRORS):
        gate.validate_gradients(gradients, inputs, protocol)


@pytest.mark.parametrize("field,value", [
    ("weighted_data_mean", "0.5"), ("weighted_physics_mean", True), ("data_gradient_norm", -1.),
    ("physics_gradient_norm", float("nan")), ("physics_to_data_norm_ratio", float("inf")),
    ("physics_to_data_norm_ratio", None), ("physics_to_data_norm_ratio", .9),
    ("gradient_cosine", 1.1), ("gradient_cosine", None),
    ("gradient_angle_degrees", 181.), ("gradient_angle_degrees", None),
    ("gradient_angle_degrees", 70.), ("total_mean", .9),
])
def test_public_gradient_finite_null_and_range_checks(bundle, field, value):
    gradients, inputs, protocol = gradient_inputs(bundle)
    gradients["grids"]["29636"]["reports"]["O-S_s7-val.json"]["records"][0][field] = value
    with pytest.raises(ERRORS):
        gate.validate_gradients(gradients, inputs, protocol)


@pytest.mark.parametrize("data_norm,physics_norm", [(0., 1.), (1., 0.), (0., 0.)])
def test_gradient_zero_norm_geometry_is_explicitly_null(bundle, data_norm, physics_norm):
    gradients, inputs, protocol = gradient_inputs(bundle)
    row = gradients["grids"]["29636"]["reports"]["O-S_s7-val.json"]["records"][0]
    row.update(data_gradient_norm=data_norm, physics_gradient_norm=physics_norm,
               physics_to_data_norm_ratio=None if data_norm == 0 else 0.,
               gradient_cosine=None, gradient_angle_degrees=None)
    gate.validate_gradients(gradients, inputs, protocol)
    row["gradient_cosine"] = 0.
    with pytest.raises(ERRORS):
        gate.validate_gradients(gradients, inputs, protocol)


def test_original_gradient_report_hash_is_not_reduced_payload_hash(bundle):
    gradients, inputs, protocol = gradient_inputs(bundle)
    public = gradients["grids"]["29636"]["reports"]["O-S_s7-val.json"]
    assert fake_hash(json.dumps(public)) != inputs["artifacts/gradient-g29636/O-S_s7-val.json"]
    gate.validate_gradients(gradients, inputs, protocol)


@pytest.mark.parametrize("label", gate.DETAIL_LABELS)
def test_every_detailed_cell_is_bound_in_source_and_its_pdf_table(bundle_template, label):
    """Exercise all 494 cells, including every endpoint/seed scalar in array cells."""
    root = bundle_template["root"]
    tex = (root / "main.tex").read_text()
    rows = bundle_template["details"][label]
    headers = ("Procedure", "Arm", "Contrast", "Representation", "Stratum", "Year")
    rendered = "\f".join("\n".join(p) for p in bundle_template["pages"])
    gate.bind_table(tex, rendered, label, rows, headers)
    block = gate.float_block(tex, label)
    count = 0
    for key, cells in rows.items():
        for j, cell in enumerate(cells):
            # Arrays contain several scalars: mutate each separately, not just the cell.
            tokens = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", cell))
            # Scientific notation is one scalar, but also test its exponent separately.
            for token in tokens:
                changed = copy.deepcopy(rows)
                changed[key][j] = cell[:token.start()] + "987.6543" + cell[token.end():]
                broken = tex.replace(block, table(label, changed, gate.float_caption(tex, label)))
                with pytest.raises(ERRORS):
                    gate.bind_table(broken, rendered, label, rows, headers)
                prefix = "  ".join(key) if isinstance(key, tuple) else key
                old = prefix + "  " + "  ".join(cells)
                new = prefix + "  " + "  ".join(changed[key])
                pages = copy.deepcopy(bundle_template["pages"])
                index = list(gate.DETAIL_LABELS).index(label) + 6
                pages[index] = [line.replace(old, new) for line in pages[index]]
                # Put the correct row in a different table: it cannot rescue this one.
                pages[5].append(old)
                broken_pdf = "\f".join("\n".join(p) for p in pages)
                with pytest.raises(ERRORS, match="PDF"):
                    gate.bind_table(tex, broken_pdf, label, rows, headers)
            count += 1
    assert count == sum(map(len, rows.values()))


def test_detailed_inventory_and_duplicate_composite_keys(bundle_template):
    details = bundle_template["details"]
    assert sum(map(len, details.values())) == 98
    assert sum(len(cells) for rows in details.values() for cells in rows.values()) == 494
    # Seven contrasts add four scalars per array-valued row; twelve intervals add one.
    assert 494 + 7 * 4 + 12 == 534
    tex = (bundle_template["root"] / "main.tex").read_text()
    rows = details["tab:seeds"]
    block = gate.float_block(tex, "tab:seeds")
    row = "D & 7 & " + " & ".join(rows[("D", "7")]) + r" \\"
    for changed in (block.replace(row, row + "\n" + row), block.replace(row, row.replace("D & 7", "D & 13")),
                    block.replace(row, "")):
        with pytest.raises(ERRORS):
            gate.bind_table(tex.replace(block, changed), "", "tab:seeds", rows, ("Procedure",))


def test_numbered_caption_and_scientific_exponent_cannot_borrow_other_rows():
    rows = {"Original float64": ["0", "8.476x10^-15", "3.326x10^-13"],
            "Float64 normalization and inverse": ["1.271x10^-15", "8.987x10^-15", "5.573x10^-13"]}
    tex = table("tab:roundtrip", rows, "Numerical round trips.")
    line = "Original float64     0     8.476 x 10     3.326 x 10-13"
    exponent = " " * line.index("     3.326") + "-15"
    pdf = "Table 1: Numerical round trips.\n\n" + exponent + "\n" + line + "\nFloat64 normalization and inverse   1.271x10-15   8.987x10-15   5.573x10-13\n"
    gate.bind_table(tex, pdf, "tab:roundtrip", rows, ("Procedure",))
    for bad in (pdf.replace(exponent, exponent.replace("-15", "-14")), pdf.replace("Table 1:", "Table 2:"),
                pdf + "\n" + pdf):
        with pytest.raises(ERRORS):
            gate.bind_table(tex, bad, "tab:roundtrip", rows, ("Procedure",))


@pytest.mark.parametrize("gap", [6, 8])
def test_two_row_pdf_scientific_layout_preserves_columns_and_original_blanks(gap):
    rows = {"Original float64": ["0", "8.476x10^-15", "3.326x10^-13"],
            "Float64 normalization and inverse": ["1.271x10^-15", "8.987x10^-15", "5.573x10^-13"]}
    tex = table("tab:roundtrip", rows, "Numerical round trips.")
    first = ("Original float64".ljust(50) + "0".ljust(24) + "8.476 \u00d7 10"
             + " " * 14 + "3.326 \u00d7 10\u221213")
    second = ("Float64 normalization and inverse".ljust(49) + "1.271 \u00d7 10"
              + " " * 15 + "8.987 \u00d7 10\u221215" + " " * 11 + "5.573 \u00d7 10\u221213")
    assert [re.search(r"\u00d7 10", line).end() for line in (first, second)] == [84, 59]
    lines = ["", " " * (84 + gap) + "\u221215", first,
             " " * (59 + gap) + "\u221215", second, "", "  "]
    assert gate.join_pdf_exponents(lines) == [
        "", first.replace("\u00d7 10", "\u00d7 10\u221215", 1),
        second.replace("\u00d7 10", "\u00d7 10\u221215", 1), "", "  "]
    caption = "Table 1: Numerical round trips.\n"
    gate.bind_table(tex, caption + "\n".join(lines), "tab:roundtrip", rows, ("Procedure",))
    for i, base_end in ((1, 84), (3, 59)):
        for exponent in (lines[i].replace("\u2212", "+"), lines[i].replace("15", "14"),
                         " " * (base_end + 9) + "\u221215", " " * (base_end - 1) + "\u221215"):
            changed = lines.copy()
            changed[i] = exponent
            with pytest.raises(ERRORS, match="PDF"):
                gate.bind_table(tex, caption + "\n".join(changed), "tab:roundtrip", rows, ("Procedure",))
    for separator in ("", "  ", "Intervening prose."):
        changed = lines[:3] + [separator] + lines[3:]
        with pytest.raises(ERRORS, match="PDF row"):
            gate.bind_table(tex, caption + "\n".join(changed), "tab:roundtrip", rows, ("Procedure",))
    with pytest.raises(ERRORS, match="orphan"):
        gate.bind_table(tex, caption + "\n".join(lines + [" " * 90 + "\u221215"]),
                        "tab:roundtrip", rows, ("Procedure",))


@pytest.mark.parametrize("lines,message", [
    (["        -15", "1x10 x10"], "ambiguous"),
    (["    -15 -15", "1x10        "], "duplicate"),
    (["        -15", "1x10  0"], "non-whitespace"),
    (["    -15", "No scientific base"], "ambiguous"),
    (["    -15", "", "1x10"], "ambiguous"),
    (["1x10-15", "    -15"], "orphan"),
])
def test_detached_pdf_exponents_refuse_ambiguous_orphan_and_cross_cell_bindings(lines, message):
    with pytest.raises(ERRORS, match=message):
        gate.join_pdf_exponents(lines)


@pytest.mark.parametrize("selection,wrong", [
    ("Raw and clipped diffusion predictions use separate validation checkpoint choices.",
     "Raw and clipped diffusion predictions use a joint validation checkpoint choice."),
    ("Validation selects checkpoints separately for raw and clipped diffusion predictions.",
     "Validation selects one shared checkpoint for raw and clipped diffusion predictions."),
])
def test_accurate_independent_and_inequality_caption_is_accepted(bundle_template, selection, wrong):
    root = bundle_template["root"]
    tex = (root / "main.tex").read_text()
    caption = gate.float_caption(tex, "tab:main")
    actual = selection + " The inequality column gives the percentage of generated bins that violate an inequality beyond rounding, excluding the equality test."
    tex = tex.replace(caption, actual)
    pages = copy.deepcopy(bundle_template["pages"])
    start = next(i for i, line in enumerate(pages[0]) if line.startswith("Table 1:"))
    end = next(i for i, line in enumerate(pages[0]) if line.startswith("D  "))
    pages[0][start:end] = ["Table 1: " + actual]
    evidence = (gate.read_json(root / "evidence/real/comparison.json"), gate.read_json(root / "evidence/real/diagnostics.json"))
    rendered = "\f".join("\n".join(p) for p in pages)
    gate.main_table(tex, rendered, evidence)
    for old, new in (("excluding the equality test", "including the equality test"),
                     (selection, wrong)):
        with pytest.raises(ERRORS):
            gate.main_table(tex.replace(old, new), rendered.replace(old, new), evidence)


def public_real_records(bundle):
    return (gate.read_json(bundle["real"] / "comparison.json"), gate.read_json(bundle["real"] / "diagnostics.json"))


@pytest.mark.parametrize("damage", ["bytes", "source", "protocol", "ids", "dtype", "transform", "extra", "object", "split"])
def test_observation_public_contract(bundle, damage):
    directory = bundle["root"] / "evidence/observations"
    path = directory / "test.npz"
    manifest = gate.read_json(directory / "MANIFEST.json")
    if damage == "bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif damage in ("source", "protocol"):
        if damage == "source":
            manifest["records"]["test.npz"]["source_record"] = "other/test.npz"
        else:
            manifest["source_protocol_sha256"] = fake_hash("other protocol")
        save(directory / "MANIFEST.json", manifest)
    else:
        with np.load(path, allow_pickle=False) as blob:
            fields = {k: blob[k] for k in blob.files}
        if damage == "ids":
            fields["ids"] = fields["ids"][::-1]
        elif damage == "dtype":
            fields["z"] = fields["z"].astype(np.float32)
        elif damage == "transform":
            fields["mean"][0] += 1
        elif damage == "extra":
            fields["alpha"] = np.zeros(480)
        elif damage == "object":
            fields["ids"] = fields["ids"].astype(object)
        else:
            fields["split"] = "TRAIN"
        np.savez(path, **fields)
        original_manifest(directory, "test.npz", "artifacts/inventory/test.npz", manifest["source_protocol_sha256"])
    with pytest.raises(ERRORS):
        gate.validate_observations(bundle["root"], public_real_records(bundle))


def test_nonzero_original_residual_and_three_distinct_roundtrip_routes(bundle_template):
    observations = gate.load_arrays(bundle_template["root"] / "evidence/observations/test.npz", "TEST")
    rows = gate.observation_roundtrips(observations)
    assert rows["Original float64"][0] == 0
    assert rows["Original float64"][1] > 0
    assert rows["Float32, six coordinates"][1] > rows["Float64 normalization and inverse"][1]
    assert rows["Float32, derived v_v"][1:] == [0, 0]
    assert rows["Float32, derived v_v"][0] == rows["Float32, six coordinates"][0] > 0


@pytest.mark.parametrize("damage", ["tertile", "feature", "ids", "means", "interval", "single_year"])
def test_test_strata_recomputed_from_public_observations(bundle_template, damage):
    root = bundle_template["root"]
    real = public_real_records(bundle_template)
    summary = real[1]
    if damage == "tertile":
        summary["tertiles"]["hidden_mean_m"]["edges"][0] += .01
    elif damage == "feature":
        summary["tertiles"]["hidden_mean_m"]["values_in_test_id_order"][0] += .01
    elif damage == "ids":
        summary["strata"]["hidden_mean_m_tertile_1"]["ids"].reverse()
    elif damage == "means":
        summary["strata"]["year_2005"]["methods"]["D_raw"]["month_balanced"] += .01
    else:
        key = "year_2005" if damage == "single_year" else "full_test"
        summary["strata"][key]["comparisons"]["O-S_raw_minus_D_raw"]["year_95"] = [0., 0.]
    observations = gate.load_arrays(root / "evidence/observations/test.npz", "TEST")
    with pytest.raises(ERRORS):
        gate.validate_test_strata(root, real, observations)


@pytest.mark.parametrize("damage", ["producer", "protocol", "input", "array", "ids", "normalizer", "feature",
                                    "membership", "mean", "empty", "cutoff", "ties", "scope", "method", "bootstrap", "private"])
def test_training_sensitivity_rebound_mutations(bundle, damage):
    root = bundle["root"]
    directory = root / "evidence/sensitivity"
    record = gate.read_json(directory / "training-strata.json")
    if damage == "producer":
        record["producer_source_sha256"]["analysis/prigm_training_strata.py"] = fake_hash("other source")
    elif damage == "protocol":
        record["protocol_sha256"] = fake_hash("other protocol")
    elif damage == "input":
        record["input_sha256"]["SELECTION.json"] = fake_hash("other selection")
    elif damage == "array":
        record["arrays"]["TRAIN"]["sha256"] = fake_hash("other TRAIN")
    elif damage == "ids":
        record["ids"].reverse()
    elif damage == "normalizer":
        record["normalization"]["b_ref_sq"] += 1
    elif damage == "feature":
        record["features_in_test_id_order"]["context_energy"][0] += 1
    elif damage == "membership":
        record["membership_in_test_id_order"]["target_energy_quartile"][0] = 0
    elif damage == "mean":
        record["groups"]["full_test"]["methods"]["D_raw"]["month_balanced"] += .01
    elif damage == "empty":
        empty = next(v for v in record["groups"].values() if v["status"] == "empty_stratum")
        empty["methods"]["D_raw"]["mean"] = 0.
    elif damage == "cutoff":
        record["cutoffs"]["target_energy"]["values"][0] = 100
    elif damage == "ties":
        record["cutoff_definitions"]["test_edge_tie_counts"]["target_energy"][0] += 1
    elif damage == "scope":
        record["groups"]["target_high"]["membership_scope"] = "confirmatory"
    elif damage == "method":
        record["methods"]["D_corrected_at_raw_checkpoint"]["selection_kind"] = "corrected"
    elif damage == "bootstrap":
        record["bootstrap"]["replicates"] = 9
    else:
        record["note"] = "/home/private/file"
    save(directory / "training-strata.json", record)
    original_manifest(directory, "training-strata.json", "artifacts/training-strata-01.json", public_real_records(bundle)[1]["protocol_sha256"])
    with pytest.raises(ERRORS):
        gate.validate_sensitivity(root, public_real_records(bundle), gate.load_arrays(root / "evidence/observations/test.npz", "TEST"))


@pytest.mark.parametrize("damage", ["last_fit", "order", "ids", "draws", "checkpoint", "selected", "array", "source",
                                    "original", "score_hash", "last_chunk", "last_window", "reduction", "metric_shape", "string", "pooled"])
def test_learned_toy_rebound_payload_mutations(bundle, damage):
    directory = bundle["root"] / "evidence/toy/learned"
    learned = gate.read_json(directory / "learned.json")
    manifest = gate.read_json(directory / "MANIFEST.json")
    last = learned["fits"][learned["fit_order"][-1]]
    if damage == "last_fit":
        learned["fits"].pop(learned["fit_order"][-1])
    elif damage == "order":
        learned["fit_order"].reverse()
    elif damage == "ids":
        learned["ids"].reverse()
    elif damage == "draws":
        learned["n_draws"] = 117
    elif damage == "checkpoint":
        last["checkpoint_sha256"] = fake_hash("other weights")
    elif damage == "selected":
        last["generation_updates"] = 4999
    elif damage == "array":
        learned["arrays"]["normalization_sha256"] = fake_hash("other normalization")
    elif damage == "source":
        learned["sources"]["inference"]["prigm/oracle.py"] = fake_hash("other source")
        manifest["sources"] = learned["sources"]
    elif damage == "original":
        learned["original_sha256"]["oracle"] = fake_hash("other oracle")
        manifest["original_sha256"] = learned["original_sha256"]
    elif damage == "score_hash":
        last["corrected"]["source_sha256"]["score"] = fake_hash("other score")
    elif damage == "last_chunk":
        key = next(k for k in reversed(manifest["input_sha256"]) if "chunk-0280" in k)
        del manifest["input_sha256"][key]
    elif damage == "last_window":
        last["corrected"]["per_window"].pop()
    elif damage == "reduction":
        last["corrected"]["secondary"]["cov90_mean"]["month_balanced"] += .01
    elif damage == "metric_shape":
        last["corrected"]["per_window"][-1]["cov90_by_position_channel"] = [0.] * 60
    elif damage == "string":
        last["corrected"]["per_window"][-1]["es5"] = "0.1"
    else:
        learned["interval_protocol"] = "pooled_117_draws"
    manifest["output_sha256"]["learned.json"] = save(directory / "learned.json", learned)
    save(directory / "MANIFEST.json", manifest)
    with pytest.raises(ERRORS):
        gate.validate_learned_toy(bundle["root"])


@pytest.mark.parametrize("decoupled", [None, False])
def test_adamw_version_metadata_preserves_decoupled_decay(bundle, decoupled):
    directory = bundle["root"] / "evidence/toy/learned"
    learned = gate.read_json(directory / "learned.json")
    optimizers = [learned["training_settings"]["optimizer"]]
    optimizers += [fit["config"]["settings"]["optimizer"] for fit in learned["fits"].values()]
    for optimizer in optimizers:
        if decoupled is None:
            optimizer.pop("decoupled_weight_decay", None)  # Older AdamW schema.
        else:
            optimizer["decoupled_weight_decay"] = decoupled
    sha = save(directory / "learned.json", learned)
    mutate(directory / "MANIFEST.json", lambda m: m["output_sha256"].update({"learned.json": sha}))
    if decoupled is None:
        gate.validate_learned_toy(bundle["root"])
    else:
        with pytest.raises(ERRORS, match="optimizer"):
            gate.validate_learned_toy(bundle["root"])


def test_atomic_reference_coverage_is_not_nominal_and_rebound_forgery_fails(bundle):
    directory = bundle["root"] / "evidence/toy/learned"
    reference = gate.read_json(directory / "reference-calibration.json")
    assert reference["secondary"]["cov50_mean"]["mean"] > .5
    assert reference["secondary"]["cov90_mean"]["mean"] > .9
    # Coherent per-window/reduced coverage still has to match the actual frozen RNG.
    for row in reference["per_window"]:
        row["cov90_mean"] = .9
        row["cov90_by_position_channel"] = np.full((12, 5), .9).tolist()
    for key, value in (("cov90_mean", .9), ("cov90_by_position_channel", np.full((12, 5), .9).tolist())):
        reference["secondary"][key] = {"mean": value, "month_balanced": value}
    sha = save(directory / "reference-calibration.json", reference)
    mutate(directory / "MANIFEST.json", lambda m: m["output_sha256"].update({"reference-calibration.json": sha}))
    with pytest.raises(ERRORS, match="same-stream"):
        gate.validate_learned_toy(bundle["root"])


@pytest.mark.parametrize("component", ["observations", "sensitivity", "toy/learned"])
def test_new_mandatory_payloads_cannot_be_omitted(bundle, component, capsys):
    (bundle["root"] / "evidence" / component / "MANIFEST.json").unlink()
    assert gate.main(["--paper-dir", str(bundle["root"])]) == 1
    assert "OVERALL: FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["[0,1]", "1x10^-5", "0.123"])
def test_numeric_arrays_cannot_hide_in_empty_header(value):
    rows = {"D": ["1.234"]}
    tex = table("tab:geometry", rows, "Fixture.")
    tex = tex.replace(r"\end{tabular}", " & " + value + r" \\\end{tabular}")
    with pytest.raises(ERRORS, match="header"):
        gate.bind_table(tex, "", "tab:geometry", rows, ("Procedure",))


def test_pdf_integer_columns_cannot_be_concatenated():
    rows = {("D", "7"): ["9560", "29636", "1.2345", "1.1234", "70.0"]}
    tex = table("tab:seeds", rows, "Fixed seed diagnostics.")
    good = "Table 1: Fixed seed diagnostics.\nD  7  9560  29636  1.2345  1.1234  70.0\n"
    gate.bind_table(tex, good, "tab:seeds", rows, ("Procedure",))
    with pytest.raises(ERRORS, match="PDF"):
        gate.bind_table(tex, good.replace("7  9560", "79  560"), "tab:seeds", rows, ("Procedure",))


def test_sensitivity_single_year_is_unavailable_not_zero_uncertainty(bundle):
    root = bundle["root"]
    directory = root / "evidence/sensitivity"
    record = gate.read_json(directory / "training-strata.json")
    features = {k: np.asarray(v) for k, v in record["features_in_test_id_order"].items()}
    # Deliberately chosen synthetic cutoff creates a singleton. This is an edge-
    # case fixture, not a claim that the unpublished TRAIN cutoffs were refitted.
    record["cutoffs"]["target_absolute_cancellation"]["values"] = [float(np.sort(features["target_absolute_cancellation"])[-2])]
    record["cutoffs"]["target_absolute_cancellation"]["training_edge_tie_counts"] = [0]
    record["cutoff_definitions"]["test_edge_tie_counts"] = {
        k: [int(np.count_nonzero(features[k] == e)) for e in v["values"]] for k, v in record["cutoffs"].items()}
    scores = gate.read_json(bundle["real"] / "scores.json")
    record.update(gate.sensitivity.summarize_groups({k: np.asarray(v) for k, v in scores["data"].items()},
                  np.asarray(scores["months"]), scores["ids"], features, record["cutoffs"]))
    single = record["groups"]["target_high"]
    assert single["status"] == "single_year" and single["n_windows"] == 1
    pair = single["comparisons"]["O-S_raw_minus_D_raw"]
    assert pair["year_95"] is None and pair["replicates_used"] == 0
    observations = gate.load_arrays(root / "evidence/observations/test.npz", "TEST")
    save(directory / "training-strata.json", record)
    original_manifest(directory, "training-strata.json", "artifacts/training-strata-01.json", record["protocol_sha256"])
    gate.validate_sensitivity(root, public_real_records(bundle), observations)
    pair["year_95"] = [0., 0.]
    save(directory / "training-strata.json", record)
    original_manifest(directory, "training-strata.json", "artifacts/training-strata-01.json", record["protocol_sha256"])
    with pytest.raises(ERRORS):
        gate.validate_sensitivity(root, public_real_records(bundle), observations)


def test_toy_clause_rejects_an_unmatched_page_top_float(bundle_template):
    root = bundle_template["root"]
    tex = (root / "main.tex").read_text()
    rendered = extracted(bundle_template["pages"])
    rendered, count = re.subn(r"On a separate set of\s+288\s+windows,",
                             "On a separate set of 288 windows,\nFigure 1: An unmatched diagnostic.\n", rendered)
    assert count == 1
    with pytest.raises(ERRORS, match="clause"):
        gate.toy_claims(tex, rendered, gate.validate_toy(root))


@pytest.mark.parametrize("damage", [None, "caption", "figure_number", "score", "sampler_name"])
def test_toy_ddim_clause_crosses_only_a_matching_operator_float(bundle_template, damage):
    root = bundle_template["root"]
    tex = (root / "main.tex").read_text()
    caption = r"A synthetic operator figure with a linear region inside $\pm0.005$."
    figure = r"\begin{figure}\caption{" + caption + r"}\label{fig:operator}\end{figure}"
    tex = tex.replace(r"\bibliography{", figure + "\n" + r"\bibliography{", 1)
    labels = re.findall(r"\\label\{(fig:[^}]+)\}", tex)
    number = labels.index("fig:operator") + 1
    rendered = extracted(bundle_template["pages"])
    pdf_caption = "A synthetic operator figure with a linear region inside ±0.005."
    insertion = f"while\n3\n\fAxis labels\nFigure {number}: {pdf_caption}\n\n"
    continuation = "25, 50, or 100 DDIM evaluations approach"
    rendered, count = re.subn(r"\s+".join(map(re.escape, continuation.split())), continuation, rendered)
    assert count == 1
    rendered, count = re.subn(r"while\s+(?=25,\s*50,\s*or\s*100 DDIM evaluations approach)",
                             lambda _: insertion, rendered)
    assert count == 1
    evidence = gate.validate_toy(root)
    if damage == "caption":
        rendered = rendered.replace(pdf_caption, pdf_caption.replace("0.005", "0.006"))
    elif damage == "figure_number":
        rendered = rendered.replace(f"Figure {number}:", f"Figure {number + 1}:")
    elif damage == "score":
        value = f"{evidence[1]['mean_es_vs_ref']:.3f}"
        rendered, count = re.subn(r"(energy score\s+)" + re.escape(value),
                                 lambda match: match[1] + "987.6543", rendered)
        assert count == 1
    elif damage == "sampler_name":
        rendered = rendered.replace("DDIM evaluations approach", "other evaluations approach")
    if damage is None:
        gate.toy_claims(tex, rendered, evidence)
    else:
        with pytest.raises(ERRORS, match="toy|clause"):
            gate.toy_claims(tex, rendered, evidence)


def test_internally_rebound_gradient_checkpoint_must_match_public_selection(bundle):
    gradients, _, _ = gradient_inputs(bundle)
    changed = fake_hash("different fixed checkpoint")
    grid = gradients["grids"]["2868"]
    grid["request"]["checkpoint_sha256"]["7"] = changed
    for split in ("train", "val"):
        grid["reports"][f"O-S_s7-{split}.json"]["checkpoint_sha256"] = changed
    save(bundle["real"] / "gradients.json", gradients)
    rebind_real(bundle, "gradients.json")
    mutate(bundle["real"] / "MANIFEST.json", lambda r: r["input_sha256"].update({
        "runs/O-S_s7/gen_00002868/state.pt": changed}))
    with pytest.raises(ERRORS, match="selected/probed checkpoint"):
        gate.validate_real(bundle["root"])


def followup_check(bundle, title):
    root = bundle["root"]
    if title.startswith("Training diffusion"):
        records = tuple(gate.read_json(root / "evidence/toy/learned" / name)
                        for name in ("learned.json", "reference-calibration.json"))
        return lambda tex, pdf: gate.learned_toy_claims(tex, pdf, records)
    if title.startswith("How the residual"):
        return lambda tex, pdf: gate.residual_weight_claims(root, tex, pdf)
    record = gate.read_json(root / "evidence/sensitivity/training-strata.json")
    return lambda tex, pdf: gate.cancellation_claims(tex, pdf, record)


def extracted(pages):
    return "\f".join("\n".join(p) for p in pages)


@pytest.mark.parametrize("row_index", range(9))
def test_every_learned_cell_and_composite_identifier_is_bound(bundle_template, row_index):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    title = "Training diffusion models on the same HMM"
    check = followup_check(b, title)
    pdf = extracted(b["pages"])
    check(tex, pdf)
    key, cells = list(b["learned_rows"].items())[row_index]
    original = [*key, *cells]
    for column in range(7):
        row = original.copy()
        row[column] = "OTHER" if column == 0 else "987.6543"
        old = " & ".join(original) + r" \\"
        new = " & ".join(row) + r" \\"
        assert old in tex
        with pytest.raises(ERRORS, match="source cells"):
            check(tex.replace(old, new) + "\nDecoy " + old, pdf)
        pages = copy.deepcopy(b["pages"])
        page = b["addition_pages"][title]
        old_pdf, new_pdf = "  ".join(original), "  ".join(row)
        assert any(old_pdf in line for line in pages[page])
        pages[page] = [line.replace(old_pdf, new_pdf) for line in pages[page]]
        pages[3].append(old_pdf)  # A correct row in another scope cannot rescue it.
        with pytest.raises(ERRORS, match="PDF row"):
            check(tex, extracted(pages))


@pytest.mark.parametrize("damage", ["duplicate", "missing", "swapped", "duplicate_id", "percent_scaling"])
def test_learned_table_order_inventory_and_percentage_scaling(bundle_template, damage):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    pdf = extracted(b["pages"])
    rows = b["learned_rows"]
    lines = [" & ".join([*k, *v]) + r" \\" for k, v in rows.items()]
    old = lines[1]
    if damage == "duplicate":
        new = old + "\n" + old
    elif damage == "missing":
        new = ""
    elif damage == "swapped":
        old, new = "\n".join(lines[1:3]), "\n".join(reversed(lines[1:3]))
    elif damage == "duplicate_id":
        new = old.replace("0.0005", "0.005", 1)
    else:
        cells = list(rows.values())[1]
        new = old.replace(" & " + cells[-1] + r" \\", f" & {float(cells[-1]) / 100:.1f}" + r" \\")
    check = followup_check(b, "Training diffusion models on the same HMM")
    with pytest.raises(ERRORS):
        check(tex.replace(old, new), pdf)


FOLLOWUP_TITLES = ("Training diffusion models on the same HMM", "How the residual penalty is weighted", "Directional cancellation with thresholds from TRAIN")


@pytest.mark.parametrize("damage", [None, "caption", "number", "value", "missing_row", "seed_sign"])
def test_cancellation_clause_rejoins_only_the_complete_annual_table(bundle_template, damage):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    labels = re.findall(r"\\label\{(tab:[^}]+)\}", tex)
    number = labels.index("tab:years") + 1
    rows = gate.source_table_rows(tex, "tab:years")
    caption = gate.clean_cell(gate.float_caption(tex, "tab:years"))
    if damage == "caption":
        caption = "A different annual population."
    elif damage == "number":
        number += 1
    elif damage == "value":
        rows[1][-1] = "987.6543"
    elif damage == "missing_row":
        rows.pop()
    insertion = f"All three seed\n18\n\fTable {number}: {caption}\n" + "\n".join("  ".join(row) for row in rows) + "\n\n"
    pdf, count = re.subn(r"All\s+three\s+seed\s+(?=differences are)",
                        lambda _: insertion, extracted(b["pages"]))
    assert count == 1
    if damage == "seed_sign":
        pdf, count = re.subn(r"differences are(\s+)negative\b", r"differences are\1positive", pdf)
        assert count == 1
    check = followup_check(b, FOLLOWUP_TITLES[2])
    if damage is None:
        check(tex, pdf)
    else:
        with pytest.raises(ERRORS, match="cancellation|clause"):
            check(tex, pdf)


@pytest.mark.parametrize("title", FOLLOWUP_TITLES)
def test_every_followup_prose_number_is_clause_bound_in_source_and_pdf(bundle_template, title):
    """Mutate each numeric token, including signs, exponents and interval endpoints."""
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    prose = b["additions"][title]
    check = followup_check(b, title)
    pdf = extracted(b["pages"])
    check(tex, pdf)
    block = gate.float_block(prose, "tab:learnedtoy") if title.startswith("Training diffusion") else ""
    spans = [(0, len(prose))]
    if block:
        start = prose.index(block)
        # Table cells have exhaustive tests above. Caption numbers belong here.
        caption = gate.float_caption(prose, "tab:learnedtoy")
        cstart = prose.index(caption)
        spans = [(0, start), (cstart, cstart + len(caption)), (start + len(block), len(prose))]
    count = 0
    for start, end in spans:
        for match in re.finditer(r"[-+]?\d+(?:\.\d+)?", prose[start:end]):
            lo, hi = start + match.start(), start + match.end()
            broken = prose[:lo] + "987.6543" + prose[hi:]
            with pytest.raises(ERRORS):
                check(tex.replace(prose, broken) + "\nDecoy " + prose, pdf)
            pages = copy.deepcopy(b["pages"])
            pages[b["addition_pages"][title]] = addition_lines(title, broken, FOLLOWUP_TITLES.index(title) + 1)
            pages[3].append(fixture_visible(prose))
            with pytest.raises(ERRORS, match="PDF"):
                check(tex, extracted(pages))
            count += 1
    assert count >= 10


@pytest.mark.parametrize("title,old,new", [
    (FOLLOWUP_TITLES[0], "clipping retains the model state and samples", "clipping selects a new model state and samples"),
    (FOLLOWUP_TITLES[0], "separately from the equality residual", "including the equality residual"),
    (FOLLOWUP_TITLES[0], "that fail a moment inequality beyond rounding tolerance", "that fail the equality by any amount"),
    (FOLLOWUP_TITLES[0], "five primary coordinates", "six generated coordinates"),
    (FOLLOWUP_TITLES[0], "tied values can raise coverage", "tied values cannot raise coverage"),
    (FOLLOWUP_TITLES[0], "would mischaracterize calibration in this toy", "fully characterize calibration in this toy"),
    (FOLLOWUP_TITLES[0], "The exact conditional ensembles in Table 2", "An independent ensemble in Table 2"),
    (FOLLOWUP_TITLES[1], "derivatives of both loss terms also matter", "loss coefficients alone determine gradient ratios"),
    (FOLLOWUP_TITLES[1], "ratio of residual to data gradient norms", "ratio of data to residual gradient norms"),
    (FOLLOWUP_TITLES[1], "three seeds, with eight windows per seed", "three seeds, with eight individual window gradients per seed"),
    (FOLLOWUP_TITLES[1], "final O-S checkpoints", "first O-S checkpoints"),
    (FOLLOWUP_TITLES[1], "first and final O-S checkpoints", "final O-S checkpoints only"),
    (FOLLOWUP_TITLES[1], "Both residuals and both estimators", "One residual and one estimator"),
    (FOLLOWUP_TITLES[1], "on TRAIN and", "on TEST and"),
    (FOLLOWUP_TITLES[1], "can accompany a small gradient", "establishes a large gradient"),
    (FOLLOWUP_TITLES[1], "batches before clipping the total gradient", "batches after clipping the total gradient"),
    (FOLLOWUP_TITLES[2], "strictly above", "at or above"),
    (FOLLOWUP_TITLES[2], "36 available bins", "36 hidden bins"),
    (FOLLOWUP_TITLES[2], "twelve hidden bins", "twelve available bins"),
    (FOLLOWUP_TITLES[2], "overlapping training windows contributes equally", "distinct training bins contributes equally"),
    (FOLLOWUP_TITLES[2], "followed inspection", "preceded inspection"),
    (FOLLOWUP_TITLES[2], "analysis is exploratory", "analysis is confirmatory"),
    (FOLLOWUP_TITLES[2], "higher energy group", "lower energy group"),
    (FOLLOWUP_TITLES[2], "A value equal to the cancellation threshold enters the rest group", "A value equal to the cancellation threshold enters the high group"),
    (FOLLOWUP_TITLES[2], "equal weight to the months present", "equal weight to individual windows"),
    (FOLLOWUP_TITLES[2], "All three seed differences are negative", "each of the three seeds has a positive difference"),
    (FOLLOWUP_TITLES[2], "assignment to these TEST groups uses the withheld values", "assignment to these TEST groups uses only available values"),
    (FOLLOWUP_TITLES[2], "lowest energy quartile", "highest energy quartile"),
    (FOLLOWUP_TITLES[2], "Interpreting them as evidence of conditional predictive superiority depends on accounting for the rule that uses withheld values to form the groups", "these losses establish superiority of a conditional distribution"),
])
def test_followup_semantic_scope_cannot_change_in_source_or_pdf(bundle_template, title, old, new):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    prose = b["additions"][title]
    assert old in prose
    broken = prose.replace(old, new)
    check = followup_check(b, title)
    with pytest.raises(ERRORS):
        check(tex.replace(prose, broken), extracted(b["pages"]))
    pages = copy.deepcopy(b["pages"])
    pages[b["addition_pages"][title]] = addition_lines(title, broken, FOLLOWUP_TITLES.index(title) + 1)
    pages[3].append(fixture_visible(prose))
    with pytest.raises(ERRORS, match="PDF"):
        check(tex, extracted(pages))


@pytest.mark.parametrize("title", FOLLOWUP_TITLES)
def test_followup_subsection_is_mandatory_in_source_and_pdf(bundle_template, title):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    pdf = extracted(b["pages"])
    check = followup_check(b, title)
    with pytest.raises(ERRORS, match="subsection"):
        check(tex.replace(r"\subsection{" + title + "}", ""), pdf)
    with pytest.raises(ERRORS, match="subsection"):
        check(tex, pdf.replace(title, "Removed heading"))


@pytest.mark.parametrize("key", ["sigma", "w"])
@pytest.mark.parametrize("factor", [2., 1. + 1e-13])
def test_rebound_gradient_schedule_must_be_shared_across_all_reports(bundle, key, factor):
    gradients, _, _ = gradient_inputs(bundle)
    # Even an early VAL report must agree; the displayed ranges use only the final grid.
    gradients["grids"]["2868"]["reports"]["O-S_s42-val.json"]["schedule"]["probe_values"][key][0] *= factor
    save(bundle["real"] / "gradients.json", gradients)
    rebind_real(bundle, "gradients.json")
    with pytest.raises(ERRORS, match="shared gradient schedule"):
        gate.residual_weight_claims(bundle["root"], (bundle["root"] / "main.tex").read_text(), extracted(bundle["pages"]))


def test_residual_weight_reread_checks_public_hash(bundle):
    path = bundle["real"] / "gradients.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ERRORS, match="hash"):
        gate.residual_weight_claims(bundle["root"], "", "")


@pytest.mark.parametrize("seed", ["7", "13", "42"])
def test_context_high_sign_claim_checks_every_seed(bundle_template, seed):
    b = bundle_template
    record = gate.read_json(b["root"] / "evidence/sensitivity/training-strata.json")
    record["groups"]["context_high"]["comparisons"]["O-S_raw_minus_D_raw"]["seed_diffs"][seed] = 0.
    with pytest.raises(ERRORS, match="all be negative"):
        gate.cancellation_claims((b["root"] / "main.tex").read_text(), extracted(b["pages"]), record)


@pytest.mark.parametrize("title", FOLLOWUP_TITLES)
def test_correct_duplicate_clause_cannot_rescue_wrong_clause_in_same_subsection(bundle_template, title):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    prose = b["additions"][title]
    anchors = ("Table compares nine fits to the stationary chain with 50 DDIM evaluations for sampling.",
               "To understand the effect on optimization, the derivatives of both loss terms also matter.",
               "Although the thresholds come from TRAIN, assignment to these TEST groups uses the withheld values. The score comparison is consequently conditional on outcome selection.")
    clause = anchors[FOLLOWUP_TITLES.index(title)]
    broken = prose.replace(clause, clause.replace("50", "51") if title.startswith("Training diffusion") else clause.replace("derivatives of both loss terms also matter", "loss coefficients alone suffice").replace("uses the withheld values", "uses only available values"))
    broken += "\n\n" + clause
    check = followup_check(b, title)
    with pytest.raises(ERRORS):
        check(tex.replace(prose, broken), extracted(b["pages"]))
    pages = copy.deepcopy(b["pages"])
    pages[b["addition_pages"][title]] = addition_lines(title, broken, FOLLOWUP_TITLES.index(title) + 1)
    with pytest.raises(ERRORS):
        check(tex, extracted(pages))


def test_missing_learned_table_fails_mandatory_inventory(bundle_template):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    assert len(gate.TABLE_LABELS) == 12
    with pytest.raises(ERRORS, match="mandatory tables"):
        gate.source_checks(b["root"], tex.replace(gate.float_block(tex, "tab:learnedtoy"), ""))


def test_clause_anchor_cannot_start_inside_another_word():
    expected = "D changes -0.0064 at fixed weights."
    anchor = gate.clause_pattern("D changes")
    decoy = "The new procedure changes the target and changes its weight. "
    gate.bind_clause((decoy + expected, decoy + expected), anchor, expected)
    for side in (0, 1):
        scopes = [expected, expected]
        scopes[side] = "And changes -0.0064 at fixed weights."
        with pytest.raises(ERRORS, match="missing or duplicate clause"):
            gate.bind_clause(scopes, anchor, expected)


@pytest.mark.parametrize("good,bad", [
    ("The interval is [-0.0750,0.0040].", "The interval is [0.0750,0.0040]."),
    ("The interval is [-0.0750,0.0040].", "The interval is [-0.0750,-0.0040]."),
    ("The interval is [-0.0750,0.0040].", "The interval is [0.0040,-0.0750]."),
    ("The weight is 2.731x10^-7.", "The weight is 2.731x10^7."),
    ("The weight is 2.731x10^-7.", "The weight is 2.731x10^+7."),
    ("The coverage is 97.49\\%.", "The coverage is 0.9749\\%."),
    ("The coverage is 97.49\\%.", "The coverage is 97.49."),
    ("The ratio is 0.1239.", "The ratio is 0.12 39."),
    ("The ratio is 1030.7.", "The ratio is 10 30.7."),
])
def test_clause_signs_exponents_percentages_and_numeric_token_boundaries(good, bad):
    gate.bind_clause((good, fixture_visible(good)), "The", good)
    with pytest.raises(ERRORS, match="source"):
        gate.bind_clause((bad, fixture_visible(good)), "The", good)
    with pytest.raises(ERRORS, match="PDF"):
        gate.bind_clause((good, fixture_visible(bad)), "The", good)


@pytest.mark.parametrize("title", FOLLOWUP_TITLES)
def test_followup_actual_pdf_mutation_fails_with_correct_text_elsewhere(bundle, title):
    prose = bundle["additions"][title]
    # A real PDF extraction, complementing the exhaustive extracted-text mutations.
    old, new = {FOLLOWUP_TITLES[0]: ("separately from the equality residual", "including the equality residual"),
                FOLLOWUP_TITLES[1]: ("batches before clipping the total gradient", "batches after clipping the total gradient"),
                FOLLOWUP_TITLES[2]: ("analysis is exploratory", "analysis is confirmatory")}[title]
    pages = copy.deepcopy(bundle["pages"])
    pages[bundle["addition_pages"][title]] = addition_lines(title, prose.replace(old, new), FOLLOWUP_TITLES.index(title) + 1)
    pages[3] += textwrap.wrap(fixture_visible(prose), width=100)
    write_pdf(bundle["root"] / "main.pdf", pages)
    with pytest.raises(ERRORS, match="PDF"):
        followup_check(bundle, title)((bundle["root"] / "main.tex").read_text(), gate.pdf_checks(bundle["root"]))


COMPANION_URL = "https://github.com/magnaprog/diffusion-consistency"


@pytest.mark.parametrize("value,valid", [
    (COMPANION_URL, True),
    ("https://github.com/magnaprog/conditional-diffusion-consistency", False),
    *[(COMPANION_URL + suffix, False) for suffix in (
        "/", "/tree/main", ".git", ".evil", "-extra", "?download=1", "#readme", "%2Fprivate", "@evil.example",
    )],
    ("x" + COMPANION_URL, False),
    ("https://evil.example/?next=" + COMPANION_URL, False),
    (COMPANION_URL.replace("https:", "http:"), False),
    (COMPANION_URL.replace("github.com", "github.com.evil.example"), False),
    (COMPANION_URL.replace("magnaprog", "MAGNAPROG"), False),
    (COMPANION_URL + " magnaprog", False),
    (COMPANION_URL + " skykev1928", False),
    (COMPANION_URL + " /home/private/study", False),
    (COMPANION_URL + r" C:\Users\private\study", False),
])
def test_companion_url_camera_ready_source_and_pdf(bundle, camera_ready_bundle, value, valid):
    tex, pages = camera_ready_bundle
    tex += "\n" + r"\url{" + value + "}"
    pages[0].append(value)
    write_pdf(bundle["root"] / "main.pdf", pages, author=gate.AUTHOR_CAMERA_READY)
    if valid:
        gate.source_checks(bundle["root"], tex)
        assert COMPANION_URL in gate.pdf_checks(bundle["root"])
    else:
        with pytest.raises(ERRORS, match="author-identifying|private"):
            gate.source_checks(bundle["root"], tex)
        with pytest.raises(ERRORS, match="author-identifying|private"):
            gate.pdf_checks(bundle["root"])


def test_companion_url_remains_private_in_anonymous_source_pdf_and_evidence(bundle):
    tex = (bundle["root"] / "main.tex").read_text() + "\n" + r"\url{" + COMPANION_URL + "}"
    with pytest.raises(ERRORS, match="author-identifying"):
        gate.source_checks(bundle["root"], tex)
    pages = copy.deepcopy(bundle["pages"])
    pages[0].append(COMPANION_URL)
    write_pdf(bundle["root"] / "main.pdf", pages)
    with pytest.raises(ERRORS, match="author-identifying"):
        gate.pdf_checks(bundle["root"])
    with pytest.raises(ERRORS):
        gate.private_check({"url": COMPANION_URL})


@pytest.mark.parametrize("camera_ready", [False, True])
@pytest.mark.parametrize("target,valid", [
    (COMPANION_URL, True), ("https://example.org/public", True),
    ("https://github.com/magnaprog/conditional-diffusion-consistency", False),
    (COMPANION_URL + ".evil", False), (COMPANION_URL + "/private", False),
    (COMPANION_URL + "?private=1", False), (COMPANION_URL + "#private", False),
    ("https://evil.example/?next=" + COMPANION_URL, False),
    ("https://github.com/magnaprog/another-repo", False),
    ("https://github.com/skykev1928/private", False), ("file:///home/private/study", False),
    (COMPANION_URL + "\n/private", False), (COMPANION_URL + "\n.evil", False),
    (COMPANION_URL + "\r/private", False), (COMPANION_URL + "\t/private", False),
    *[(COMPANION_URL + suffix, False) for suffix in (
        "/", ".git", "%2Fprivate", "@evil.example", " ", "\n", "\r", "\t", "\x00",
    )],
    ("\n" + COMPANION_URL, False),
    (COMPANION_URL.replace("magnaprog", "MAGNAPROG"), False),
    ("https://evil.example/?next=\n   1  Annotation    " + COMPANION_URL, False),
])
def test_companion_url_uri_annotations_allow_only_exact_camera_ready_target(bundle, request, camera_ready, target, valid):
    if camera_ready:
        _, pages = request.getfixturevalue("camera_ready_bundle")
        author = gate.AUTHOR_CAMERA_READY
    else:
        pages = copy.deepcopy(bundle["pages"])
        author = ""
    # The label contains no identity: only the real PDF URI target can reject this leak.
    pages[0].append("Companion code")
    write_pdf(bundle["root"] / "main.pdf", pages, author=author, uri=target)
    if valid and (camera_ready or target != COMPANION_URL):
        gate.pdf_checks(bundle["root"])
    else:
        with pytest.raises(ERRORS, match="author-identifying|private"):
            gate.pdf_checks(bundle["root"])


@pytest.mark.parametrize("extra_uri", [
    None, COMPANION_URL + ".evil", COMPANION_URL + "\n/private", "file:///home/private/study",
])
def test_companion_url_structured_uri_objects_check_every_link(bundle, camera_ready_bundle, extra_uri):
    _, pages = camera_ready_bundle
    path = bundle["root"] / "main.pdf"
    write_pdf(path, pages, author=gate.AUTHOR_CAMERA_READY, uri=COMPANION_URL)
    with pikepdf.open(path, allow_overwriting_input=True) as document:
        annotation = document.pages[0].Annots[0]
        annotation.A = document.make_indirect(annotation.A)
        if extra_uri is not None:
            # A second page's target must still be checked beside the allowed URI.
            document.pages[1].Annots = pikepdf.Array([pikepdf.Dictionary(
                Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Link,
                Rect=pikepdf.Array([30, 900, 850, 920]),
                A=pikepdf.Dictionary(S=pikepdf.Name.URI, URI=pikepdf.String(extra_uri)),
            )])
        document.save(object_stream_mode=pikepdf.ObjectStreamMode.generate)
    if extra_uri is None:
        gate.pdf_checks(bundle["root"])
    else:
        with pytest.raises(ERRORS, match="author-identifying|private"):
            gate.pdf_checks(bundle["root"])


def test_companion_url_does_not_exempt_pdf_metadata(bundle, camera_ready_bundle):
    _, pages = camera_ready_bundle
    write_pdf(bundle["root"] / "main.pdf", pages, author=gate.AUTHOR_CAMERA_READY,
              subject=COMPANION_URL, uri=COMPANION_URL)
    with pytest.raises(ERRORS, match="author-identifying"):
        gate.pdf_checks(bundle["root"])


@pytest.mark.parametrize("camera_ready", [False, True])
@pytest.mark.parametrize("base,target,valid", [
    (None, "README.html", True), ("", "README.html", True),
    ("https://example.org/public/", "README.html", True),
    ("https://example.org/public/", COMPANION_URL, True),
    ("https://github.com/magnaprog/private/", "README.html", False),
    ("file:///home/private/study/", "README.html", False),
    (COMPANION_URL, "README.html", False),
    (COMPANION_URL, None, False),
    ("https://github.com/magnaprog/private/", COMPANION_URL, False),
    ("file:///home/private/study/", None, False),
    (COMPANION_URL + "\n/private", "README.html", False),
    ("https://example.org/public/", COMPANION_URL + "\n", False),
    ("file:///", "home/private", False),
    ("file:///", "Users/private", False),
    ("file:///", "tmp/private", False),
    ("file:///", "public/README.html", True),
    ("file:///", "ho\nme/private", False),
    ("https://example.org/public/", COMPANION_URL.replace("magnaprog", "mag\nnaprog"), False),
])
def test_pdf_uri_base_has_no_companion_exception(bundle, request, camera_ready, base, target, valid):
    if camera_ready:
        _, pages = request.getfixturevalue("camera_ready_bundle")
        author = gate.AUTHOR_CAMERA_READY
    else:
        pages, author = copy.deepcopy(bundle["pages"]), ""
    path = bundle["root"] / "main.pdf"
    write_pdf(path, pages, author=author, uri=target)
    with pikepdf.open(path, allow_overwriting_input=True) as document:
        document.Root.URI = pikepdf.Dictionary()
        if base is not None:
            document.Root.URI.Base = pikepdf.String(base)
        document.save()
    if valid and (camera_ready or target != COMPANION_URL):
        gate.pdf_checks(bundle["root"])
    else:
        with pytest.raises(ERRORS, match="author-identifying|private"):
            gate.pdf_checks(bundle["root"])


@pytest.mark.parametrize("field,value", [
    ("URI", "https://example.org/"), ("URI", 42), ("URI", pikepdf.Array()),
    ("Base", 42), ("Base", pikepdf.Name.Invalid),
    ("Base", pikepdf.Array()), ("Base", pikepdf.Dictionary()),
])
def test_pdf_uri_base_rejects_malformed_values(bundle, field, value):
    path = bundle["root"] / "main.pdf"
    with pikepdf.open(path, allow_overwriting_input=True) as document:
        document.Root.URI = value if field == "URI" else pikepdf.Dictionary(Base=value)
        document.save()
    with pytest.raises(ERRORS, match="PDF catalog URI must be a dictionary|PDF URI base must be a string"):
        gate.pdf_checks(bundle["root"])


@pytest.mark.parametrize("email,valid", [(gate.EMAIL_CAMERA_READY, True), ("other@example.org", False)])
def test_pdf_uri_base_checks_camera_ready_email(bundle, camera_ready_bundle, email, valid):
    _, pages = camera_ready_bundle
    path = bundle["root"] / "main.pdf"
    write_pdf(path, pages, author=gate.AUTHOR_CAMERA_READY, uri=COMPANION_URL)
    with pikepdf.open(path, allow_overwriting_input=True) as document:
        document.Root.URI = pikepdf.Dictionary(Base=pikepdf.String("mailto:" + email))
        document.save()
    if valid:
        gate.pdf_checks(bundle["root"])
    else:
        with pytest.raises(ERRORS, match="undeclared email"):
            gate.pdf_checks(bundle["root"])


@pytest.mark.parametrize("scope", ["source", "PDF"])
def test_bad_gradient_timing_cannot_hide_behind_valid_duplicate(bundle_template, scope):
    b = bundle_template
    title = FOLLOWUP_TITLES[1]
    tex = (b["root"] / "main.tex").read_text()
    prose = b["additions"][title]
    clause = ("To examine how the penalty enters optimization, the gradient comparison evaluates the weighted data and residual terms on the same batches before clipping the total gradient.")
    assert prose.count(clause) == 1
    changed = prose.replace(clause, clause.replace("before", "after") + "\n" + clause, 1)
    check = followup_check(b, title)
    check(tex, extracted(b["pages"]))
    pages = copy.deepcopy(b["pages"])
    if scope == "source":
        tex = tex.replace(prose, changed, 1)
    else:
        pages[b["addition_pages"][title]] = addition_lines(title, changed, FOLLOWUP_TITLES.index(title) + 1)
    with pytest.raises(ERRORS, match=scope + ": missing or duplicate clause"):
        check(tex, extracted(pages))


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("valid_duplicate", [False, True])
@pytest.mark.parametrize("prefix,old,new", [
    ("This property also holds numerically", "40 eligible intervals", "41 eligible intervals"),
    ("This property also holds numerically", "2.35", "2.36"),
    ("Replacing each bin by its mean vector", "0.1235", "0.2235"),
])
def test_q15_complete_claim_rejects_wrong_value_and_correct_decoy(
        bundle_template, scope, valid_duplicate, prefix, old, new):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    rendered = gate.prose_text(extracted(b["pages"]))
    real = gate.validate_real(b["root"])
    gate.data_audit(b["root"], tex, rendered, real)
    q15 = next(p for p in tex.split("\n\n") if p.startswith("This property also holds numerically"))
    clause = q15.split("Replacing each bin", 1)[0].strip() if prefix.startswith("This") else "Replacing each bin" + q15.split("Replacing each bin", 1)[1]
    if scope == "PDF":
        clause, old, new = (gate.prose_text(v, source=True) for v in (clause, old, new))
    original = tex if scope == "source" else rendered
    assert original.count(clause) == 1 and clause.count(old) == 1
    changed = original.replace(clause, clause.replace(old, new, 1), 1)
    if valid_duplicate:
        changed += "\n" + clause
    with pytest.raises(ERRORS, match=scope + ": (clause differs|missing or duplicate clause)"):
        gate.data_audit(b["root"], changed if scope == "source" else tex,
                       changed if scope == "PDF" else rendered, real)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("valid_duplicate", [False, True])
@pytest.mark.parametrize("value", ["1000", "900"])
def test_reintroduced_training_minute_claim_stays_checked(bundle_template, scope, valid_duplicate, value):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    rendered = extracted(b["pages"])
    real = gate.validate_real(b["root"])
    # Absence is permitted after relocation to the release guide.
    gate.data_audit(b["root"], tex, rendered, real)
    clause = "The training dates contain 1000 recorded minutes, including 900 with valid magnetic vectors."
    original = tex if scope == "source" else rendered
    valid = original + "\n" + clause
    gate.data_audit(b["root"], valid if scope == "source" else tex,
                   valid if scope == "PDF" else rendered, real)
    changed = original + "\n" + clause.replace(value, str(int(value) + 1), 1)
    if valid_duplicate:
        changed += "\n" + clause
    with pytest.raises(ERRORS, match="clause differs|missing or duplicate clause"):
        gate.data_audit(b["root"], changed if scope == "source" else tex,
                       changed if scope == "PDF" else rendered, real)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("valid_duplicate", [False, True])
@pytest.mark.parametrize("alter_value", [False, True])
def test_exact_coverage_binds_original_ensemble_table(bundle_template, scope, valid_duplicate, alter_value):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    check = followup_check(b, FOLLOWUP_TITLES[0])
    rendered = extracted(b["pages"])
    check(tex, rendered)
    original = tex if scope == "source" else rendered
    match = re.search(r"The exact conditional ensembles in Table\s+2[\s\S]*?calibration in this toy\.", original)
    assert match is not None
    clause = match[0]
    wrong = re.sub(r"Table\s+2", "Table 99", clause, count=1)
    if alter_value:
        wrong, n = re.subn(r"(coverage\s+of\s+)\d+\.\d+", r"\g<1>99.9", wrong, count=1)
        assert n == 1
    changed = original[:match.start()] + wrong + original[match.end():]
    if valid_duplicate:
        # Add inside the same subsection, before the next heading.
        changed = changed[:match.start()] + clause + "\n" + changed[match.start():]
    with pytest.raises(ERRORS, match=scope + ": (clause differs|missing or duplicate clause)"):
        check(changed if scope == "source" else tex, changed if scope == "PDF" else rendered)


@pytest.mark.parametrize("scope", ["source", "PDF"])
@pytest.mark.parametrize("value", ["999", "-999", "NaN"])
@pytest.mark.parametrize("valid_duplicate", [False, True])
@pytest.mark.parametrize("kind", ["toy", "gradient", "learned", "learned_caption",
                                  "cancellation_train", "cancellation_groups", "cancellation_seeds"])
def test_numeric_claim_slots_reject_malformed_values_and_valid_duplicates(
        bundle_template, scope, value, valid_duplicate, kind):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    rendered = extracted(b["pages"])
    if kind == "toy":
        evidence = gate.validate_toy(b["root"])
        check = lambda source, pdf: gate.toy_claims(source, pdf, evidence)
        pattern = r"Across\s+(?P<slot>36)\s+replicate contexts[\s\S]*?independent exact conditional samples\."
    elif kind == "gradient":
        check = followup_check(b, FOLLOWUP_TITLES[1])
        pattern = r"At\s+\$?t\s*=\s*(?P<slot>25)\$?,\s+the ranges[\s\S]*?(?=\s+Each range spans)"
    elif kind.startswith("cancellation"):
        check = followup_check(b, FOLLOWUP_TITLES[2])
        pattern = {
            "cancellation_train": r"Every one of the (?P<slot>\d+) overlapping training windows[\s\S]*?distinct bins\.",
            "cancellation_groups": r"The\s+(?P<slot>\d+)\s+groups\s+comprise[\s\S]*?each\s+group\.",
            "cancellation_seeds": r"All (?P<slot>three) seed differences[\s\S]*? zero\.",
        }[kind]
    else:
        check = followup_check(b, FOLLOWUP_TITLES[0])
        pattern = (r"compares\s+(?P<slot>nine)\s+fits[\s\S]*?for sampling\." if kind == "learned"
                   else r"(?P<slot>Nine)\s+diffusion fits[\s\S]*?seed 7\.")
    check(tex, rendered)
    original = tex if scope == "source" else rendered
    matches = list(re.finditer(pattern, original))
    assert len(matches) == 1
    match = matches[0]
    clause = match[0]
    start, end = match.start("slot") - match.start(), match.end("slot") - match.start()
    wrong = clause[:start] + value + clause[end:]
    changed = original[:match.start()] + wrong + (" " + clause if valid_duplicate else "") + original[match.end():]
    message = "missing or duplicate clause" if valid_duplicate else "clause differs"
    with pytest.raises(ValueError, match=scope + ": " + message):
        check(changed if scope == "source" else tex, changed if scope == "PDF" else rendered)



@pytest.mark.parametrize("scope", ["source", "PDF", "both"])
@pytest.mark.parametrize("prefix", ["-", "+", "NaN-"])
@pytest.mark.parametrize("valid_duplicate", [False, True])
def test_caption_fit_count_keeps_malformed_prefix_in_claim(
        bundle_template, scope, prefix, valid_duplicate):
    b = bundle_template
    tex = (b["root"] / "main.tex").read_text()
    rendered = extracted(b["pages"])
    check = followup_check(b, FOLLOWUP_TITLES[0])
    check(tex, rendered)
    values = [tex, rendered]
    for side in (0, 1):
        if scope not in ("both", ("source", "PDF")[side]):
            continue
        matches = list(re.finditer(r"Nine\s+diffusion\s+fits[\s\S]*?seed\s+7\.", values[side]))
        assert len(matches) == 1
        match = matches[0]
        clause = match[0]
        changed = prefix + clause + (" " + clause if valid_duplicate else "")
        values[side] = values[side][:match.start()] + changed + values[side][match.end():]
    location = "PDF" if scope == "PDF" else "source"
    message = "missing or duplicate clause" if valid_duplicate else "clause differs"
    with pytest.raises(ValueError, match=location + ": " + message):
        check(*values)
