"""Check evidence export with locally constructed study records.

The expensive checks of the complete primary-score and supplementary-metric
inputs are mocked. Source identities, completion records, allowed paths,
statistical calculations, serialization and output creation remain active.
The supplementary-input checker has separate coverage in
test_prigm_followup_summary.py.

Gradient records contain invented values in the probe's required format;
they are not newly calculated gradients. Checkpoint checks validate the
hashes of locally committed synthetic files without fitting models. These
tests verify export behavior and do not replace scientific evidence.
"""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import export_prigm_paper_evidence as export
from experiments.run_prigm_repair import base_protocol
from prigm.artifacts import json_bytes
from prigm.recovery import ContractMismatch, PublicationError, sha256_file


def save(path, record):
    path.write_bytes(json_bytes(record))
    return sha256_file(path)


def change_record(path, mutate):
    row = json.loads(path.read_bytes())
    mutate(row)
    return save(path, row)


def gradient_report(root, protocol, checkpoint, seed, split, ids):
    """Small actual-schema record with absolute producer paths."""
    identity = protocol["arrays"][split]
    indices = np.linspace(0, len(ids) - 1, 8, dtype=np.int64).tolist()
    selected = [ids[i] for i in indices]
    contract = checkpoint["contract"]
    return {
        "schema_version": 1, "diagnostic": "weighted_physics_data_parameter_gradients",
        "scope": "Local checkpoint diagnostic on fixed TRAIN or VAL windows; not causal evidence for observed score changes.",
        "limitations": ["Gradients of weighted batch means, not means of per-window gradient norms."],
        "gradient_definition": "Full parameter vector; Euclidean norms, dot-product cosine and angle in degrees; float64 reductions.",
        "undefined": "Ratio is null when the data norm is zero; cosine and angle are null when either norm is zero. No epsilon denominators.",
        "run_root": str(root / f"runs/O-S_s{seed}"), "generation_updates": checkpoint["generation"],
        "checkpoint_sha256": checkpoint["files"]["state.pt"], "training_contract": contract,
        "training_source_hashes": contract["source_hashes"],
        "current_source_hashes": protocol["inference_source_hashes"],
        "source_comparison": "Training source is recorded, not required to equal current probe source.",
        "model": {"class": "prigm.model.ConditionalUNet", "widths": contract["widths"],
                  "n_generated": 6, "n_parameters": 1603398, "mode": "eval"},
        "arrays_path": str(root / f"artifacts/inventory/{split.lower()}.npz"), "arrays": identity,
        "selection": {"rule": "np.linspace(0, n_available - 1, min(n_available, n_windows), dtype=int64); first index if count is one",
                      "n_requested": 8, "n_selected": 8, "indices": indices,
                      "ids": selected, "ids_sha256": export.id_hash(selected)},
        "inputs": {"u0_sha256": "a" * 64, "observed_sha256": "b" * 64,
                   "state_convention": "clean_observed_flanks_hidden_diffusion", "hidden_slice": [18, 30]},
        "noise": {"seed": 7003, "generator": "torch.Generator", "device": "cpu", "dtype": "torch.float32",
                  "shape": [8, 6, 48], "sha256": "c" * 64,
                  "reuse": "One full noise tensor shared across all timesteps, estimators and residuals; only the gap is corrupted."},
        "normalization": {"mean": [0.] * 6, "scale": [1.] * 6, "b_ref_sq": 1., "n_bins": 48,
                          "sha256": identity["normalization_sha256"]},
        "c_r": .0005, "schedule": {**contract["settings"]["diffusion"], "probe_timesteps": [1, 25, 50, 100],
                                  "probe_values": {k: [.9996843093705424, .5, .1, 2.4285728691465365e-07]
                                                   for k in ("a", "s", "w", "sigma")}},
        "environment": {"python": "3.11.15", "numpy": "2.4.3", "torch": "2.6.0+cu124", "platform": "Linux",
                        "device": "cpu", "model_dtype": "torch.float32", "torch_threads": 2,
                        "torch_interop_threads": 16, "deterministic_algorithms": False, "mkldnn_enabled": True,
                        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        "records": [{"t": t, "estimator": e, "residual": r, "weighted_data_mean": .125,
                     "weighted_physics_mean": .0625, "total_mean": .1875,
                     "data_gradient_norm": .5590513193835821, "physics_gradient_norm": .3128004208417973,
                     "physics_to_data_norm_ratio": .5595200476169084, "gradient_cosine": .1074481167758676,
                     "gradient_angle_degrees": 83.83176852991537}
                    for t in (1, 25, 50, 100) for e in ("M", "S") for r in ("obs", "inst")],
    }


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "private-study-name"
    inventory = root / "artifacts/inventory"
    inventory.mkdir(parents=True)
    ids = [f"{year}-{month:02d}-{day:02d}-s52" for year in export.bootstrap.YEARS
           for month, day in ((8, 1), (8, 2), (9, 1))]
    months = np.array([int(w[:4]) * 100 + int(w[5:7]) for w in ids])
    z = np.ones((len(ids), 48, 6))
    z[:, 18:30, 3] = np.arange(len(ids))[:, None]
    for split in ("TRAIN", "VAL", "TEST"):
        np.savez(inventory / f"{split.lower()}.npz", z=z, ids=np.asarray(ids), split=split,
                 mean=np.zeros(6), scale=np.ones(6), b_ref_sq=1., n_bins=48)
    protocol = base_protocol(root, inventory)
    # Private execution metadata is omitted, never rewritten into a new protocol.
    protocol["private_execution_path"] = "/home/fixture-owner/private-run"
    digest = save(root / "PROTOCOL.json", protocol)
    selection_hash = save(root / "SELECTION.json", {"fixture": "selected fixed models"})
    phase_hash = save(root / "TEST_STARTED.json", {"protocol_sha256": digest,
                      "selection_sha256": selection_hash, "previous_test_exposure": True})
    rng = np.random.default_rng(31)
    data = {f"{arm}_{kind}": 1 + rng.random((3, len(ids))) for arm in export.bootstrap.ARMS
            for kind in (*export.bootstrap.KINDS, "corrected_at_raw_checkpoint")}
    data.update({f"{name}_{kind}": 1 + rng.random((1, len(ids)))
                 for name in export.bootstrap.CONTROLS for kind in export.bootstrap.KINDS})
    primary = root / "artifacts/synthetic-primary.json"
    save(primary, {label: values.tolist() for label, values in data.items()})
    secondary = root / "artifacts/synthetic-secondaries.json"
    save(secondary, {"fixture": "full secondary validation is mocked"})
    hashes = {str(p): sha256_file(p) for p in (root / "PROTOCOL.json", root / "SELECTION.json",
              root / "TEST_STARTED.json", inventory / "train.npz", inventory / "test.npz", primary)}
    methods = {}
    for label, values in data.items():
        control = len(values) == 1
        kind = "raw" if label.endswith("_raw") else "corrected"
        keys = ["shared"] if control else ["7", "13", "42"]
        meta = {"procedure": label.split("_")[0] if control else "diffusion",
                "scoring_kind": kind, "fixed_model_seeds": [] if control else [7, 13, 42],
                "score_files": {k: str(primary) for k in keys}}
        if control:
            meta["n_baseline_realizations"] = 1
        else:
            meta.update(selection_kind="raw" if label.endswith("raw_checkpoint") else kind,
                        generations={k: 2868 for k in keys})
        methods[label] = meta
    provenance = {"study_root": str(root), "protocol_sha256": digest,
                  "selection_sha256": selection_hash, "test_started_sha256": phase_hash,
                  "arrays": protocol["arrays"]["TEST"], "ids": ids, "months": months.tolist(),
                  "input_sha256": hashes, "source_sha256": protocol["inference_source_hashes"],
                  "methods": methods, "selected": {"D_s7": {"raw": {
                      "generation": 2868, "checkpoint_sha256": "e" * 64, "score": 1.2}}}}
    calls = {"load": 0, "summary": 0}

    def load(actual_root, actual_hash):
        assert actual_root == root and actual_hash == digest
        calls["load"] += 1
        return data, months, provenance

    monkeypatch.setattr(export.bootstrap, "load_study", load)
    # Use the real producer, not a test copy of the exporter's metadata/statistics.
    export.bootstrap.main(["--study-root", str(root), "--protocol-sha256", digest,
                           "--output", str(root / export.COMPARISON)])
    report = {"schema_version": 1, "analysis_source_sha256": sha256_file(Path(export.diagnostics.__file__)),
              **{k: provenance[k] for k in ("ids", "months", "arrays", "protocol_sha256",
                                            "selection_sha256", "test_started_sha256")},
              "input_sha256": {Path(p).relative_to(root).as_posix(): h for p, h in hashes.items()},
              "methods": {label: {"n_realizations": len(values), "metrics": {
                  "es5": {k: v.tolist() for k, v in export.diagnostics.reductions(values.mean(axis=0), months).items()}},
                  "seedwise": {key: {"es5": {k: v.tolist() for k, v in export.diagnostics.reductions(row, months).items()}}
                               for key, row in zip(meta["score_files"], values)}}
                  for (label, values), meta in zip(data.items(), methods.values())},
              **export.diagnostics.energy_strata(data, months, ids, z, replicates=13)}
    report["input_sha256"]["artifacts/synthetic-secondaries.json"] = sha256_file(secondary)
    # Deliberately noncanonical whitespace: diagnostics must be copied byte-for-byte.
    (root / export.SUMMARY).write_text(json.dumps(report, indent=1) + "\n\n")

    def summarize(actual_root, actual_hash):
        assert actual_root == root and actual_hash == digest
        calls["summary"] += 1
        return copy.deepcopy(report)

    monkeypatch.setattr(export.diagnostics, "summarize_study", summarize)
    for name in ("run_fixed_followup.py", "run_cpu_after_real.py", "run_probe_grid.py"):
        (root / name).write_bytes(b"# synthetic launcher fixture\n")
    gradient_hashes = {}
    for generation in (2868, 29636):
        parent = root / f"artifacts/gradient-g{generation}"
        parent.mkdir()
        outputs, checkpoints = {}, {}
        for seed in (7, 13, 42):
            fixed = root / f"runs/O-S_s{seed}/gen_{generation:08d}"
            fixed.mkdir(parents=True)
            (fixed / "state.pt").write_bytes(f"synthetic checkpoint {seed} {generation}".encode())
            contract = {
                "contract_version": 2, "arm": "O-S", "seed": seed, "c_r": .0005,
                "n_generated": 6, "n_aux": 0, "widths": [64, 128, 128], "n_examples": len(ids),
                "batch_size": 128, "max_updates": 29636, "estimator": "S", "residual": "obs",
                "source_hashes": protocol["training_source_hashes"], "data_hashes": protocol["training_data_hashes"],
                "normalization_hash": protocol["arrays"]["TRAIN"]["normalization_sha256"],
                "settings": {"milestones": protocol["milestones"], "chronological": True,
                             "diffusion": {"steps": 100, "cosine_offset": .008, "beta_cap": .999,
                                           "snr_clip": 5., "sigma_1": "sigma_2",
                                           "array_hashes": {k: "d" * 64 for k in ("a", "s", "w", "sigma", "betas", "alpha_bar", "snr")}}},
            }
            checkpoint = {"generation": generation, "updates": generation, "contract": contract,
                          "files": {"state.pt": sha256_file(fixed / "state.pt")}}
            save(fixed / "COMMIT.json", checkpoint)
            checkpoints[str(seed)] = checkpoint["files"]["state.pt"]
            for split in ("TRAIN", "VAL"):
                filename = f"O-S_s{seed}-{split.lower()}.json"
                outputs[filename] = save(parent / filename, gradient_report(root, protocol, checkpoint, seed, split, ids))
        request = save(parent / "REQUEST.json", {
            "protocol_sha256": digest, "generation": generation, "arm": "O-S", "seeds": [7, 13, 42],
            "splits": ["TRAIN", "VAL"], "n_windows_per_split": 8, "noise_seed": 7003, "c_r": .0005,
            "checkpoint_sha256": checkpoints, "launcher_sha256": sha256_file(root / "run_probe_grid.py"),
            "selection": "All three O-S seeds at the fixed first and final milestones; no score-based checkpoint or window selection.",
            "interpretation": "Local weighted-gradient geometry, not evidence that it caused a predictive difference.",
        })
        gradient_hashes[str(generation)] = save(parent / "COMPLETED.json", {
            "request_sha256": request, "outputs": outputs})
    save(root / "PIPELINE_COMPLETED.json", {
        "protocol_sha256": digest, "comparison_sha256": sha256_file(root / export.COMPARISON),
        "selection_sha256": selection_hash, "launcher_sha256": sha256_file(root / "run_fixed_followup.py")})
    save(root / "CPU_FOLLOWUP_COMPLETED.json", {
        "protocol_sha256": digest, "summary_sha256": sha256_file(root / export.SUMMARY),
        "analysis_source_sha256": report["analysis_source_sha256"],
        "launcher_sha256": sha256_file(root / "run_cpu_after_real.py"),
        "gradient_completion_sha256": gradient_hashes})
    calls.update(load=0, summary=0)
    return SimpleNamespace(root=root, digest=digest, data=data, months=months, provenance=provenance,
                           report=report, calls=calls, output=tmp_path / "public")


def run(study):
    return export.export_study(study.root, study.digest, study.output)


def rebind(study, relative):
    receipt, key = ("PIPELINE_COMPLETED.json", "comparison_sha256") if relative == export.COMPARISON else (
        "CPU_FOLLOWUP_COMPLETED.json", "summary_sha256")
    change_record(study.root / receipt, lambda row: row.update({key: sha256_file(study.root / relative)}))


def rebind_gradient(study, generation, filename):
    parent = study.root / f"artifacts/gradient-g{generation}"
    if filename == "REQUEST.json":
        change_record(parent / "COMPLETED.json", lambda row: row.update(request_sha256=sha256_file(parent / filename)))
    elif filename != "COMPLETED.json":
        change_record(parent / "COMPLETED.json", lambda row: row["outputs"].update({filename: sha256_file(parent / filename)}))
    change_record(study.root / "CPU_FOLLOWUP_COMPLETED.json", lambda row: row["gradient_completion_sha256"].update(
        {str(generation): sha256_file(parent / "COMPLETED.json")}))


def test_cli_publishes_original_numbers_bytes_and_figure_compatible_manifest(study, capsys):
    before = {p: sha256_file(p) for p in study.root.rglob("*") if p.is_file()}
    capsys.readouterr()
    assert export.main(["--study-root", str(study.root), "--protocol-sha256", study.digest,
                        "--output-dir", str(study.output)]) == 0
    assert study.calls == {"load": 1, "summary": 1}
    manifest = json.loads((study.output / "MANIFEST.json").read_bytes())
    assert set(p.name for p in study.output.iterdir()) == {
        "comparison.json", "diagnostics.json", "scores.json", "protocol-summary.json", "gradients.json", "MANIFEST.json"}
    assert manifest["exporter_source_sha256"] == sha256_file(Path(export.__file__))
    assert "Reduced anonymous numeric evidence" in manifest["scope"]
    assert "Not the original execution protocol" in manifest["scope"]
    for name, digest in manifest["output_sha256"].items():
        assert sha256_file(study.output / name) == digest
    for name, digest in manifest["input_sha256"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert sha256_file(study.root / name) == digest
    assert len(manifest["original_sha256"]) == 5
    assert (study.output / "diagnostics.json").read_bytes() == (study.root / export.SUMMARY).read_bytes()
    original = json.loads((study.root / export.COMPARISON).read_bytes())
    public = json.loads((study.output / "comparison.json").read_bytes())
    for key in ("arm_means", "seed_means", "comparisons", "exposure", "n_replicates", "seed",
                "fixed_training_seeds", "multiplicity", "year_blocks", "two_year_blocks", "month_counts",
                "interpretation", "comparison_scope", "uncertainty_scope", "selected", "protocol_sha256",
                "selection_sha256", "test_started_sha256"):
        assert public[key] == original[key]
    assert public["source_comparison_sha256"] == sha256_file(study.root / export.COMPARISON)
    scores = json.loads((study.output / "scores.json").read_bytes())
    assert scores["schema_version"] == 1 and scores["fixed_training_seeds"] == [7, 13, 42]
    assert scores["ids"] == study.provenance["ids"] and scores["months"] == study.months.tolist()
    assert len(scores["data"]) == 18
    for label, values in study.data.items():
        assert scores["data"][label] == values.tolist()
        assert scores["row_ids"][label] == (["shared"] if len(values) == 1 else ["7", "13", "42"])
    rebuilt = export.bootstrap.compare({k: np.asarray(v) for k, v in scores["data"].items()}, np.asarray(scores["months"]))
    for key, value in rebuilt.items():
        assert public[key] == value
    protocol = json.loads((study.output / "protocol-summary.json").read_bytes())
    assert protocol["source_protocol_sha256"] == study.digest and protocol["schema_version"] == 1
    assert "array_root" not in protocol and "private_execution_path" not in protocol
    cli = json.loads(capsys.readouterr().out)
    assert cli == {"manifest": "MANIFEST.json", "n_outputs": 5}
    gradients = json.loads((study.output / "gradients.json").read_bytes())
    assert set(gradients) == {"schema_version", "protocol_sha256", "scope", "grids"}
    assert gradients["schema_version"] == 1 and gradients["protocol_sha256"] == study.digest
    assert gradients["scope"] == export.GRADIENT_SCOPE
    assert set(gradients["grids"]) == {"2868", "29636"}
    for generation, grid in gradients["grids"].items():
        parent = study.root / f"artifacts/gradient-g{generation}"
        assert set(grid) == {"request", "source_request_sha256", "source_completion_sha256", "reports"}
        assert grid["request"] == json.loads((parent / "REQUEST.json").read_bytes())
        assert grid["source_request_sha256"] == sha256_file(parent / "REQUEST.json")
        assert grid["source_completion_sha256"] == sha256_file(parent / "COMPLETED.json")
        assert set(grid["reports"]) == {f"O-S_s{s}-{split}.json" for s in (7, 13, 42) for split in ("train", "val")}
        for filename, row in grid["reports"].items():
            original_row = json.loads((parent / filename).read_bytes())
            assert Path(original_row["run_root"]).is_absolute()
            assert Path(original_row["arrays_path"]).is_absolute()
            assert row == {**original_row,
                           "run_root": export.diagnostics._relative(original_row["run_root"], study.root),
                           "arrays_path": export.diagnostics._relative(original_row["arrays_path"], study.root)}
            assert {key for key in row if row[key] != original_row[key]} == {"run_root", "arrays_path"}
            assert manifest["input_sha256"][f"artifacts/gradient-g{generation}/{filename}"] == sha256_file(parent / filename)
            assert manifest["input_sha256"][f"artifacts/gradient-g{generation}/{filename}"] != export.sha256_bytes(json_bytes(row))
            assert len(row["records"]) == 16 and row["records"][0]["data_gradient_norm"] == .5590513193835821
    assert manifest["input_sha256"]["run_probe_grid.py"] == sha256_file(study.root / "run_probe_grid.py")
    for text in [p.read_text() for p in study.output.iterdir()] + [json.dumps(cli)]:
        assert str(study.root) not in text and "/home/" not in text and "fixture-owner" not in text
    assert {p: sha256_file(p) for p in before} == before
    path = Path(export.__file__).resolve().parents[1] / "paper/workshops/prigm/fig_audit.py"
    spec = importlib.util.spec_from_file_location("prigm_audit_export_test", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    record, _, rows = audit.load_comparison(study.output / "comparison.json", manifest["output_sha256"]["comparison.json"])
    assert len(rows) == 7 and record == public


@pytest.mark.parametrize("damage", ["arm_mean", "seed_mean", "mean_diff", "year_ci", "two_year_ci",
                                    "relative_ci", "flag", "integer_flag", "method", "exposure",
                                    "n_replicates", "month_counts", "interpretation"])
def test_rebound_comparison_tampering_is_not_validated_by_summary_means(study, damage):
    def mutate(row):
        pair = row["comparisons"]["O-S_raw_minus_D_raw"]
        if damage == "arm_mean":
            row["arm_means"]["O-S_raw"] += .01
        elif damage == "seed_mean":
            row["seed_means"]["D_raw"]["7"] += .01
        elif damage == "mean_diff":
            pair["mean_diff"] += .01
        elif damage in ("year_ci", "two_year_ci", "relative_ci"):
            pair[{"year_ci": "year_95", "two_year_ci": "twoyear_95", "relative_ci": "rel_year_95"}[damage]][0] += .01
        elif damage == "flag":
            pair["descriptive_gates"]["both_difference_intervals_below_zero"] = not pair["descriptive_gates"]["both_difference_intervals_below_zero"]
        elif damage == "integer_flag":
            pair["seeds_agree_neg"] = int(pair["seeds_agree_neg"])
        elif damage == "method":
            pair["method"] = "D_raw"
        elif damage == "exposure":
            row["exposure"] = "blind"
        elif damage == "n_replicates":
            row["n_replicates"] = 10000.0
        elif damage == "month_counts":
            row["month_counts"]["200508"] += 1
        else:
            row["interpretation"] = "This is blind confirmation."
    change_record(study.root / export.COMPARISON, mutate)
    rebind(study, export.COMPARISON)
    with pytest.raises(ContractMismatch):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("damage", ["metric", "seedwise", "stratum_ci", "stratum_flag", "ids", "extra_key"])
def test_entire_diagnostic_record_is_compared(study, damage):
    def mutate(row):
        if damage == "metric":
            row["methods"]["D_raw"]["metrics"]["es5"]["mean"] += .01
        elif damage == "seedwise":
            row["methods"]["D_raw"]["seedwise"]["7"]["es5"]["mean"] += .01
        elif damage == "stratum_ci":
            row["strata"]["full_test"]["comparisons"]["O-S_raw_minus_D_raw"]["year_95"][0] += .01
        elif damage == "stratum_flag":
            row["strata"]["full_test"]["status"] = "empty_stratum"
        elif damage == "ids":
            row["ids"][0], row["ids"][1] = row["ids"][1], row["ids"][0]
        else:
            row["new_field"] = "unvalidated"
    change_record(study.root / export.SUMMARY, mutate)
    rebind(study, export.SUMMARY)
    with pytest.raises(ContractMismatch):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("relative", ["PIPELINE_COMPLETED.json", "CPU_FOLLOWUP_COMPLETED.json",
                                      "artifacts/gradient-g2868/COMPLETED.json", "artifacts/gradient-g29636/COMPLETED.json"])
def test_missing_completion_refuses_before_gates(study, relative):
    (study.root / relative).unlink()
    with pytest.raises(FileNotFoundError):
        run(study)
    assert study.calls == {"load": 0, "summary": 0} and not study.output.exists()


@pytest.mark.parametrize("relative", [export.COMPARISON, export.SUMMARY, "SELECTION.json",
                                      "run_fixed_followup.py", "run_cpu_after_real.py", "run_probe_grid.py",
                                      "artifacts/gradient-g2868/O-S_s7-train.json",
                                      "artifacts/gradient-g2868/COMPLETED.json",
                                      "artifacts/gradient-g29636/REQUEST.json",
                                      "artifacts/gradient-g29636/O-S_s42-val.json"])
def test_stale_receipt_payload_hashes_refuse(study, relative):
    path = study.root / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ContractMismatch, match="hash"):
        run(study)
    assert not study.output.exists()


def test_wrong_summary_source_and_changed_prigm_source_refuse(study, monkeypatch):
    change_record(study.root / "CPU_FOLLOWUP_COMPLETED.json", lambda row: row.update(analysis_source_sha256="a" * 64))
    with pytest.raises(ContractMismatch):
        run(study)
    change_record(study.root / "CPU_FOLLOWUP_COMPLETED.json", lambda row: row.update(
        analysis_source_sha256=sha256_file(Path(export.diagnostics.__file__))))
    original = export.require_source_match

    def mismatch(protocol):
        altered = copy.deepcopy(protocol)
        altered["inference_source_hashes"]["prigm/run_bootstrap.py"] = "a" * 64
        original(altered)

    monkeypatch.setattr(export, "require_source_match", mismatch)
    with pytest.raises(ContractMismatch, match="source changed"):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("damage", ["swapped_ids", "duplicate_control", "missing_method"])
def test_loader_matrix_and_population_contract_is_retained(study, damage):
    if damage == "swapped_ids":
        study.provenance["ids"][0], study.provenance["ids"][1] = study.provenance["ids"][1], study.provenance["ids"][0]
    elif damage == "duplicate_control":
        study.data["gaussian_raw"] = np.repeat(study.data["gaussian_raw"], 3, axis=0)
    else:
        del study.data["G5_corrected_at_raw_checkpoint"]
    with pytest.raises(ContractMismatch):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("private", ["See /home/unknown/person.", "See /tmp/private/file.",
                                     "C:\\Users\\someone\\file", "fixture-owner", "private-study-name",
                                     "person@example.org", "access_token", "../outside.json", "password=abc"])
def test_safe_summary_requires_safe_keys_and_values_even_when_recomputed(study, private):
    study.report["note"] = {private: private}
    save(study.root / export.SUMMARY, study.report)
    rebind(study, export.SUMMARY)
    with pytest.raises(ContractMismatch, match="unsafe private"):
        run(study)
    assert not study.output.exists()


def test_private_scientific_protocol_text_is_rejected_not_redacted(study):
    # This directly exercises the public-record policy without weakening protocol pins.
    with pytest.raises(ContractMismatch, match="unsafe private"):
        export._safe({"selection": "chosen by fixture-owner"}, {"fixture-owner"})


@pytest.mark.parametrize("kind", ["external", "symlink"])
def test_input_containment_cannot_be_bypassed(study, kind):
    external = study.root.parent / "outside.json"
    external.write_bytes(b"{}")
    path = external
    if kind == "symlink":
        path = study.root / "escape.json"
        path.symlink_to(external)
    study.provenance["input_sha256"][str(path)] = sha256_file(external)
    # Match original provenance so the actual containment check owns the refusal.
    change_record(study.root / export.COMPARISON, lambda row: row.update(input_sha256=study.provenance["input_sha256"]))
    rebind(study, export.COMPARISON)
    with pytest.raises(ContractMismatch, match="escapes study root"):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling_symlink", "missing_parent"])
def test_outputs_must_be_fresh_with_existing_parent(study, kind):
    if kind == "directory":
        study.output.mkdir()
        (study.output / "partial.json").write_bytes(b"preserve")
    elif kind == "file":
        study.output.write_bytes(b"preserve")
    elif kind in ("symlink", "dangling_symlink"):
        study.output.symlink_to(study.root if kind == "symlink" else study.root / "absent")
    else:
        study.output = study.output / "missing-parent" / "public"
    with pytest.raises(FileNotFoundError if kind == "missing_parent" else FileExistsError):
        run(study)
    assert study.calls == {"load": 0, "summary": 0}
    if kind == "directory":
        assert (study.output / "partial.json").read_bytes() == b"preserve"
    elif kind == "file":
        assert study.output.read_bytes() == b"preserve"
    elif "symlink" in kind:
        assert study.output.is_symlink()


def test_output_mkdir_race_preserves_other_writer(study, monkeypatch):
    original = export.diagnostics.summarize_study

    def race(*args):
        result = original(*args)
        study.output.mkdir()
        (study.output / "other.json").write_bytes(b"other writer")
        return result

    monkeypatch.setattr(export.diagnostics, "summarize_study", race)
    with pytest.raises(FileExistsError):
        run(study)
    assert {p.name for p in study.output.iterdir()} == {"other.json"}
    assert (study.output / "other.json").read_bytes() == b"other writer"


@pytest.mark.parametrize("relative", ["artifacts/synthetic-primary.json", "run_probe_grid.py",
                                      "artifacts/gradient-g2868/REQUEST.json", "artifacts/gradient-g29636/O-S_s7-val.json",
                                      "runs/O-S_s7/gen_00002868/state.pt", "runs/O-S_s7/gen_00002868/COMMIT.json",
                                      "artifacts/inventory/val.npz"])
@pytest.mark.parametrize("timing", ["before_mkdir", "after_outputs"])
def test_input_rehash_prevents_completion_and_preserves_partial_directory(study, monkeypatch, timing, relative):
    primary = study.root / relative
    if timing == "before_mkdir":
        original = export.diagnostics.summarize_study

        def mutate(*args):
            result = original(*args)
            primary.write_bytes(primary.read_bytes() + b" ")
            return result

        monkeypatch.setattr(export.diagnostics, "summarize_study", mutate)
    else:
        original = export.atomic_write_bytes

        def mutate(path, payload):
            original(path, payload)
            if path.name == "protocol-summary.json":
                primary.write_bytes(primary.read_bytes() + b" ")

        monkeypatch.setattr(export, "atomic_write_bytes", mutate)
    with pytest.raises(ContractMismatch, match="input changed"):
        run(study)
    assert not (study.output / "MANIFEST.json").exists()
    if timing == "before_mkdir":
        assert not study.output.exists()
    else:
        assert len(list(study.output.iterdir())) == 5
        before = {p: p.read_bytes() for p in study.output.iterdir()}
        with pytest.raises(FileExistsError):
            run(study)
        assert {p: p.read_bytes() for p in before} == before


def test_write_failure_leaves_partial_uncompleted_and_manifest_is_last(study, monkeypatch):
    original = export.atomic_write_bytes
    writes = []

    def fail(path, payload):
        writes.append(path.name)
        if len(writes) == 3:
            raise OSError("synthetic write failure")
        original(path, payload)

    monkeypatch.setattr(export, "atomic_write_bytes", fail)
    with pytest.raises(OSError, match="synthetic write failure"):
        run(study)
    assert writes == ["comparison.json", "diagnostics.json", "scores.json"]
    assert len(list(study.output.iterdir())) == 2 and not (study.output / "MANIFEST.json").exists()
    with pytest.raises(FileExistsError):
        run(study)


def test_tolerance_preserves_original_numbers_instead_of_replacing_them(study):
    change_record(study.root / export.COMPARISON,
                  lambda row: row["arm_means"].update(D_raw=row["arm_means"]["D_raw"] + 5e-13))
    rebind(study, export.COMPARISON)
    original = json.loads((study.root / export.COMPARISON).read_bytes())
    run(study)
    public = json.loads((study.output / "comparison.json").read_bytes())
    assert public["arm_means"] == original["arm_means"]
    assert public["arm_means"] != export.bootstrap.compare(study.data, study.months)["arm_means"]


@pytest.mark.parametrize("left,right", [(True, 1.), (1, True), (1., 1), ([1.], [1., 2.]),
                                         ({"x": 1.}, {"y": 1.}), (float("nan"), 1.),
                                         (1., float("inf")), (1. + 3e-12, 1.)])
def test_recursive_comparison_is_type_shape_key_and_finiteness_strict(left, right):
    with pytest.raises(ContractMismatch):
        export._agree(left, right)


@pytest.mark.parametrize("text", ["note[/private/array]", "see ../outside.json", "by_fixture-owner",
                                  "private_api_key", "AWS_SECRET_ACCESS_KEY", "author_name"])
def test_private_strings_cannot_hide_in_punctuation_or_compound_keys(text):
    with pytest.raises(ContractMismatch, match="unsafe private"):
        export._safe({text: "value"}, {"fixture-owner"})


def test_absolute_and_relative_float_tolerances_are_both_used():
    export._agree(5e-13, 0.)
    export._agree(1e6 + 5e-7, 1e6)
    with pytest.raises(ContractMismatch):
        export._agree(2e-12, 0.)


@pytest.mark.parametrize("relative", ["PIPELINE_COMPLETED.json", "CPU_FOLLOWUP_COMPLETED.json"])
def test_late_completion_change_leaves_no_manifest(study, monkeypatch, relative):
    original = export.atomic_write_bytes

    def mutate(path, payload):
        original(path, payload)
        if path.name == "protocol-summary.json":
            receipt = study.root / relative
            receipt.write_bytes(receipt.read_bytes() + b" ")

    monkeypatch.setattr(export, "atomic_write_bytes", mutate)
    with pytest.raises(ContractMismatch, match="input changed"):
        run(study)
    assert len(list(study.output.iterdir())) == 5 and not (study.output / "MANIFEST.json").exists()


@pytest.mark.parametrize("which", ["exporter", "summary"])
def test_late_source_change_leaves_no_manifest_without_editing_sources(study, monkeypatch, which):
    original_write, original_hash = export.atomic_write_bytes, export.sha256_file
    target = Path(export.__file__ if which == "exporter" else export.diagnostics.__file__).resolve()
    changed = False

    def write(path, payload):
        nonlocal changed
        original_write(path, payload)
        if path.name == "protocol-summary.json":
            changed = True

    def digest(path):
        return "a" * 64 if changed and Path(path).resolve() == target else original_hash(path)

    monkeypatch.setattr(export, "atomic_write_bytes", write)
    monkeypatch.setattr(export, "sha256_file", digest)
    with pytest.raises(ContractMismatch, match="source changed during export"):
        run(study)
    assert not (study.output / "MANIFEST.json").exists()


@pytest.mark.parametrize("kind", ["file_collision", "directory_swap", "output_corruption"])
def test_publication_preserves_collisions_and_refuses_replaced_outputs(study, monkeypatch, kind):
    original = export.atomic_write_bytes
    preserved = study.output.with_name("preserved-partial")

    def interfere(path, payload):
        original(path, payload)
        if path.name != "comparison.json":
            return
        if kind == "file_collision":
            (study.output / "diagnostics.json").write_bytes(b"other writer")
        elif kind == "directory_swap":
            study.output.rename(preserved)
            study.output.symlink_to(preserved, target_is_directory=True)
        else:
            path.write_bytes(b"changed output")

    monkeypatch.setattr(export, "atomic_write_bytes", interfere)
    with pytest.raises((FileExistsError, ContractMismatch)):
        run(study)
    assert not (study.output / "MANIFEST.json").exists()
    if kind == "file_collision":
        assert (study.output / "diagnostics.json").read_bytes() == b"other writer"
    elif kind == "directory_swap":
        assert study.output.is_symlink() and (preserved / "comparison.json").is_file()


def test_duplicate_json_keys_and_nonfinite_numbers_refuse(study):
    path = study.root / export.COMPARISON
    payload = path.read_bytes()
    path.write_bytes(payload.replace(b'{', b'{"schema_version": 2,', 1))
    rebind(study, export.COMPARISON)
    with pytest.raises(ContractMismatch, match="duplicate JSON key"):
        run(study)
    path.write_bytes(payload)
    row = json.loads(payload)
    row["arm_means"]["D_raw"] = float("nan")
    path.write_text(json.dumps(row))
    rebind(study, export.COMPARISON)
    with pytest.raises(ContractMismatch, match="numeric value differs"):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("field,value", [
    ("protocol_sha256", "a" * 64), ("arm", "I-S"), ("generation", 2868), ("generation", 29636.),
    ("seeds", [7, 13]), ("seeds", [7, 42, 13]), ("seeds", [7, 13, 42.]),
    ("splits", ["TRAIN", "TEST"]), ("splits", ["train", "val"]),
    ("n_windows_per_split", 7), ("n_windows_per_split", 8.), ("noise_seed", 7004),
    ("c_r", .005), ("checkpoint_sha256", {"7": "a" * 64}), ("launcher_sha256", "b" * 64),
])
def test_rebound_gradient_request_semantics_refuse(study, field, value):
    change_record(study.root / "artifacts/gradient-g29636/REQUEST.json", lambda row: row.update({field: value}))
    rebind_gradient(study, 29636, "REQUEST.json")
    with pytest.raises(ContractMismatch):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("path,value", [
    (("schema_version",), 1.), (("diagnostic",), "unweighted_gradients"),
    (("generation_updates",), 2868), (("generation_updates",), 29636.),
    (("checkpoint_sha256",), "a" * 64), (("c_r",), .005),
    (("current_source_hashes", "prigm/run_gradient_probe.py"), "b" * 64),
    (("training_source_hashes", "prigm/run_fit.py"), "b" * 64),
    (("training_contract", "arm"), "I-S"), (("training_contract", "seed"), 13),
    (("training_contract", "estimator"), "M"), (("training_contract", "residual"), "inst"),
    (("training_contract", "settings", "chronological"), 1),
    (("arrays", "sha256"), "c" * 64), (("arrays", "ids_sha256"), "d" * 64),
    (("arrays", "split"), "TEST"), (("arrays", "n_windows"), 8),
    (("arrays", "normalization_sha256"), "d" * 64),
    (("arrays_path",), "artifacts/inventory/train.npz"), (("arrays_path",), "../outside.npz"),
    (("run_root",), "runs/O-S_s13"), (("noise", "seed"), 7004), (("noise", "shape"), [7, 6, 48]),
    (("selection", "n_selected"), 7), (("selection", "n_selected"), 8.),
    (("selection", "n_requested"), 9), (("selection", "indices"), list(range(8))),
    (("selection", "ids_sha256"), "a" * 64), (("selection", "rule"), "lowest energy windows"),
    (("inputs", "hidden_slice"), [17, 30]), (("model", "n_generated"), 5),
    (("model", "n_parameters"), True), (("model", "mode"), "train"),
    (("normalization", "scale"), [2.] * 6), (("schedule", "probe_timesteps"), [1, 25, 50]),
    (("schedule", "probe_values", "a"), [1.]), (("environment", "mkldnn_enabled"), 1),
])
def test_rebound_gradient_report_identity_and_subset_refuse(study, path, value):
    def mutate(row):
        for key in path[:-1]:
            row = row[key]
        row[path[-1]] = value
    change_record(study.root / "artifacts/gradient-g29636/O-S_s7-val.json", mutate)
    rebind_gradient(study, 29636, "O-S_s7-val.json")
    with pytest.raises(ContractMismatch):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "estimator", "residual", "t", "float_t", "bool_t"])
def test_gradient_records_must_exactly_cover_the_cartesian_grid(study, damage):
    def mutate(row):
        records = row["records"]
        if damage == "missing":
            records.pop()
        elif damage == "extra":
            records.append(records[0])
        elif damage == "duplicate":
            records[-1] = records[0]
        else:
            key, value = {"estimator": ("estimator", "D"), "residual": ("residual", "none"),
                          "t": ("t", 99), "float_t": ("t", 1.), "bool_t": ("t", True)}[damage]
            records[0][key] = value
    change_record(study.root / "artifacts/gradient-g2868/O-S_s42-train.json", mutate)
    rebind_gradient(study, 2868, "O-S_s42-train.json")
    with pytest.raises(ContractMismatch, match="Cartesian"):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("damage", ["indices_and_ids", "ids_and_hash", "missing_source", "missing_report", "extra_report"])
def test_gradient_self_consistent_subset_or_incomplete_receipt_is_not_sufficient(study, damage):
    filename = "COMPLETED.json" if "report" in damage else "O-S_s7-val.json"
    def mutate(row):
        if damage == "missing_report":
            row["outputs"].pop("O-S_s42-train.json")
        elif damage == "extra_report":
            row["outputs"]["O-S_s7-test.json"] = "a" * 64
        elif damage == "missing_source":
            row["current_source_hashes"].pop("prigm/run_gradient_probe.py")
        else:
            if damage == "indices_and_ids":
                row["selection"]["indices"] = list(range(8))
                row["selection"]["ids"] = study.provenance["ids"][:8]
            else:
                row["selection"]["ids"][0] = "2005-08-unselected"
            row["selection"]["ids_sha256"] = export.id_hash(row["selection"]["ids"])
    change_record(study.root / f"artifacts/gradient-g29636/{filename}", mutate)
    rebind_gradient(study, 29636, filename)
    with pytest.raises(ContractMismatch):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("field,value", [
    ("data_gradient_norm", float("nan")), ("physics_gradient_norm", float("inf")),
    ("weighted_data_mean", float("-inf")), ("total_mean", None), ("weighted_physics_mean", True),
    ("data_gradient_norm", -1.), ("physics_to_data_norm_ratio", float("inf")),
    ("physics_to_data_norm_ratio", None), ("gradient_cosine", None),
    ("gradient_angle_degrees", None), ("gradient_cosine", 1.1), ("gradient_angle_degrees", 181.),
    ("schedule_array", float("nan")), ("normalization_array", float("inf")),
    ("extra_array", float("nan")),
])
def test_gradient_nonfinite_or_invalid_geometry_refuses(study, field, value):
    path = study.root / "artifacts/gradient-g29636/O-S_s7-val.json"
    row = json.loads(path.read_bytes())
    if field == "schedule_array":
        row["schedule"]["probe_values"]["a"][0] = value
    elif field == "normalization_array":
        row["normalization"]["mean"][0] = value
    elif field == "extra_array":
        row["extra_numeric_array"] = [value]
    else:
        row["records"][0][field] = value
    path.write_text(json.dumps(row))  # Permit deliberately invalid JSON numbers in the source fixture.
    rebind_gradient(study, 29636, path.name)
    with pytest.raises((ContractMismatch, ValueError)):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("data_norm,physics_norm,ratio", [(0., 1., None), (1., 0., 0.), (0., 0., None)])
def test_legitimately_undefined_geometry_and_all_extra_numbers_are_preserved(study, data_norm, physics_norm, ratio):
    path = study.root / "artifacts/gradient-g29636/O-S_s7-val.json"
    def mutate(row):
        row["records"][0].update(data_gradient_norm=data_norm, physics_gradient_norm=physics_norm,
                                 physics_to_data_norm_ratio=ratio, gradient_cosine=None, gradient_angle_degrees=None)
        row["records"].reverse()  # Coverage, not producer ordering, is required.
        row["extra_numeric_array"] = [1.2345678901234567, -2.4285728691465365e-07]
        row["records"][0]["extra_scalar"] = 9.876543210987654
    change_record(path, mutate)
    rebind_gradient(study, 29636, path.name)
    run(study)
    original = json.loads(path.read_bytes())
    public = json.loads((study.output / "gradients.json").read_bytes())["grids"]["29636"]["reports"][path.name]
    assert public == {**original, "run_root": "runs/O-S_s7", "arrays_path": "artifacts/inventory/val.npz"}


@pytest.mark.parametrize("replacement", ["earlier_generation", "unbound_hash"])
def test_request_and_report_agreement_cannot_replace_fixed_checkpoint(study, replacement):
    digest = (sha256_file(study.root / "runs/O-S_s7/gen_00002868/state.pt")
              if replacement == "earlier_generation" else "e" * 64)
    parent = study.root / "artifacts/gradient-g29636"
    change_record(parent / "REQUEST.json", lambda row: row["checkpoint_sha256"].update({"7": digest}))
    rebind_gradient(study, 29636, "REQUEST.json")
    for split in ("train", "val"):
        filename = f"O-S_s7-{split}.json"
        change_record(parent / filename, lambda row: row.update(checkpoint_sha256=digest))
        rebind_gradient(study, 29636, filename)
    with pytest.raises(ContractMismatch):
        run(study)
    assert not study.output.exists()


def test_actual_fixed_checkpoints_are_validated_without_latest_selection_or_gradients(study, monkeypatch):
    import torch
    calls = []
    original = export.verify_checkpoint
    def verify(root, name, generation, protocol):
        calls.append((name, generation))
        return original(root, name, generation, protocol)
    def forbidden(*args, **kwargs):
        raise AssertionError("export must not compute gradients")
    monkeypatch.setattr(export, "verify_checkpoint", verify)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    for seed in (7, 13, 42):
        (study.root / f"runs/O-S_s{seed}/LATEST").write_text("99999999\n")
    manifest = run(study)
    assert set(calls) == {(f"O-S_s{s}", g) for s in (7, 13, 42) for g in (2868, 29636)} and len(calls) == 6
    assert all("LATEST" not in path for path in manifest["input_sha256"])
    for name, generation in calls:
        for filename in ("COMMIT.json", "state.pt"):
            path = f"runs/{name}/gen_{generation:08d}/{filename}"
            assert manifest["input_sha256"][path] == sha256_file(study.root / path)


@pytest.mark.parametrize("damage", ["payload", "contract"])
def test_actual_fixed_checkpoint_gate_refuses_even_with_matching_reports(study, damage):
    fixed = study.root / "runs/O-S_s7/gen_00029636"
    if damage == "payload":
        (fixed / "state.pt").write_bytes(b"tampered checkpoint")
    else:
        change_record(fixed / "COMMIT.json", lambda row: row["contract"].update(residual="inst"))
        for split in ("train", "val"):
            filename = f"O-S_s7-{split}.json"
            change_record(study.root / f"artifacts/gradient-g29636/{filename}",
                          lambda row: row["training_contract"].update(residual="inst"))
            rebind_gradient(study, 29636, filename)
    with pytest.raises((ContractMismatch, PublicationError)):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("private", ["private_note", "private_request"])
def test_gradient_privacy_is_not_weakened_to_preserve_report_fields(study, private):
    filename = "REQUEST.json" if private == "private_request" else "O-S_s7-val.json"
    def mutate(row):
        row["note"] = "See /home/fixture-owner/private-run."
    change_record(study.root / f"artifacts/gradient-g29636/{filename}", mutate)
    rebind_gradient(study, 29636, filename)
    with pytest.raises(ContractMismatch, match="unsafe private"):
        run(study)
    assert not study.output.exists()


@pytest.mark.parametrize("damage", ["collision", "corruption", "symlink"])
def test_fifth_output_is_hash_checked_and_never_overwritten(study, monkeypatch, damage):
    original = export.atomic_write_bytes
    def interfere(path, payload):
        original(path, payload)
        if damage == "collision" and path.name == "protocol-summary.json":
            (study.output / "gradients.json").write_bytes(b"preserve other writer")
        elif path.name == "gradients.json":
            if damage == "corruption":
                path.write_bytes(b"tampered gradients output")
            elif damage == "symlink":
                path.unlink()
                path.symlink_to(study.root / "artifacts/gradient-g29636/O-S_s7-val.json")
    monkeypatch.setattr(export, "atomic_write_bytes", interfere)
    with pytest.raises((ContractMismatch, FileExistsError)):
        run(study)
    assert not (study.output / "MANIFEST.json").exists()
    if damage == "collision":
        assert (study.output / "gradients.json").read_bytes() == b"preserve other writer"
    before = {p: p.read_bytes() for p in study.output.iterdir()}
    with pytest.raises(FileExistsError):
        run(study)
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("argv", [[], ["--study-root", "unused", "--output-dir", "unused"]])
def test_cli_requires_explicit_protocol_pin(argv):
    with pytest.raises(SystemExit):
        export.main(argv)
