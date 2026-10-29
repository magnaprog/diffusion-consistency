"""Check paper-bundle contents and packaging rules with local synthetic files.

The fixture supplies invented evidence, file hashes and a substitute verifier
so tests can exercise the packager's inclusion and identity checks. Some
figure files contain only format signatures; they are not usable scientific
figures. The tests do not compile TeX or establish that a released paper
reproduces scientific results.
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import zipfile

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_prigm_paper", REPO / "scripts/package_prigm_paper.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def figure(paper, stem="hmm", input_name="evidence/toy/hmm-stationary-01.json"):
    directory = paper / "figures" / stem
    directory.mkdir(parents=True)
    generator = paper / f"fig_{stem}.py"
    generator.write_text('"""Standalone anonymous figure generator fixture."""\n')
    pdf, png = directory / f"fig_{stem}.pdf", directory / f"fig_{stem}.png"
    # Magic bytes suffice for packager tests; these are not compilable figures.
    pdf.write_bytes(b"%PDF-1.4\nsynthetic figure\n%%EOF\n")
    png.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic figure\n")
    save_json(directory / f"fig_{stem}.manifest.json", {
        "schema_version": 2 if stem == "hmm" else 1, "figure": f"fig_{stem}",
        "source": {"path": generator.name, "sha256": sha(generator.read_bytes())},
        "inputs": {"record": {"path": Path(input_name).name,
                              "sha256": sha((paper / input_name).read_bytes())}},
        "output_sha256": {path.name: sha(path.read_bytes()) for path in (pdf, png)},
    })


@pytest.mark.parametrize("damage", [None, "missing_binding", "changed_style"])
def test_shared_plot_style_is_copied_and_bound(paper, damage):
    """A generator's shared presentation input belongs in its archive."""
    style = paper / "prigm.mplstyle"
    style.write_text("font.family: Nimbus Roman\n")
    source = paper / "fig_hmm.py"
    source.write_text(source.read_text() + '\nSTYLE = "prigm.mplstyle"\n')
    path = paper / "figures/hmm/fig_hmm.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source"]["sha256"] = sha(source.read_bytes())
    manifest["style"] = {"path": style.name, "sha256": sha(style.read_bytes())}
    if damage == "missing_binding":
        del manifest["style"]
    elif damage == "changed_style":
        style.write_text("font.family: sans-serif\n")
    save_json(path, manifest)
    if damage:
        with pytest.raises(ValueError, match="shared-style SHA256"):
            packager.collect(paper)
    else:
        files, _, _ = packager.collect(paper)
        assert files[style.name] == style.read_bytes()


@pytest.fixture
def paper(tmp_path, monkeypatch):
    root = tmp_path / "paper"
    root.mkdir()
    style = (REPO / "paper/workshops/prigm/neurips_2025.sty").read_bytes()
    assert sha(style) == packager.STYLE_SHA256
    (root / "neurips_2025.sty").write_bytes(style)
    (root / "main.tex").write_text(
        r"\documentclass{article}" "\n"
        r"\PassOptionsToPackage{numbers, sort&compress}{natbib}" "\n"
        r"\usepackage[dblblindworkshop]{neurips_2025}" "\n"
        r"\makeatletter" "\n" + packager.NOTICE + "\n" + r"\makeatother" "\n"
        r"\usepackage{graphicx,hyperref}" "\n"
        r"\hypersetup{hidelinks,pdfauthor={},pdftitle={A controlled comparison}}" "\n"
        r"\title{A controlled comparison}" "\n"
        r"\author{Anonymous}" "\n"
        r"\begin{document}\maketitle" "\n"
        r"A complete synthetic fixture. \citep{public}" "\n"
        r"\includegraphics[width=\linewidth]{figures/hmm/fig_hmm.pdf}" "\n"
        r"\bibliography{references}\bibliographystyle{unsrtnat}" "\n"
        r"\end{document}" "\n"
    )
    # Surnames in public citations must not trigger the concrete identity scan.
    (root / "references.bib").write_text(
        "@article{public, author={Lee, Jane and March, John}, title={Public results}, year={2025}}\n")
    toy = root / "evidence/toy"
    toy.mkdir(parents=True)
    np.savez(toy / "toy.npz", x=np.arange(12).reshape(3, 4), labels=np.array(["test", "validation"]))
    toy_hash = sha((toy / "toy.npz").read_bytes())
    metadata = {"schema_version": 2, "toy_eval_sha256": toy_hash,
                "boundary_law": "stationary", "conditioning": "pooled"}
    for name in packager.TOY_FILES - {"toy.npz"}:
        save_json(toy / name, {**metadata, "mean": 1.25})
    learned = toy / "learned"
    learned.mkdir()
    originals = {"protocol": sha(b"toy protocol"), "array": toy_hash,
                 "oracle": sha((toy / "oracle-stationary-01.json").read_bytes()),
                 "trained": sha(b"toy trained"), "evaluated": sha(b"toy evaluated")}
    sources = {"training": {"prigm/train.py": sha(b"training source")},
               "inference": {"prigm/run_eval.py": sha(b"inference source")},
               "coordinator_sha256": sha(b"coordinator source")}
    toy_identity = {"sha256": toy_hash, "split": "TOY", "n_windows": 3,
                    "ids_sha256": sha(b"toy IDs"), "normalization_sha256": sha(b"toy normalization")}
    save_json(learned / "learned.json", {
        "schema_version": 1, "original_sha256": originals, "sources": sources,
        "arrays": toy_identity, "bounded_fixture": True})
    save_json(learned / "reference-calibration.json", {
        "schema_version": 1, "source_oracle_sha256": originals["oracle"],
        "source_protocol_sha256": originals["protocol"], "arrays": toy_identity,
        "oracle_metadata": metadata, "bounded_fixture": True})
    save_json(learned / "MANIFEST.json", {
        "schema_version": 1, "original_sha256": originals, "sources": sources,
        "exporter_source_sha256": sha(b"exporter source"),
        "input_sha256": {"PROTOCOL.json": originals["protocol"], "artifacts/toy.npz": toy_hash,
                         "TRAINED.json": originals["trained"], "EVALUATED.json": originals["evaluated"]},
        "output_sha256": {n: sha((learned / n).read_bytes())
                          for n in ("learned.json", "reference-calibration.json")}})
    observations = root / "evidence/observations"
    observations.mkdir()
    np.savez(observations / "test.npz", z=np.zeros((2, 48, 6)),
             ids=np.array(["2018-01-01", "2018-02-01"]), split=np.array("TEST"),
             mean=np.zeros(6), scale=np.ones(6), b_ref_sq=np.array(1.), n_bins=np.array(48))
    protocol_hash = sha(b"real protocol")
    identities = {split: {"sha256": sha(split.encode()), "split": split, "n_windows": 2,
                         "ids_sha256": sha((split + " IDs").encode()),
                         "normalization_sha256": sha(b"real normalization")}
                  for split in ("TRAIN", "VAL", "TEST")}
    identities["TEST"]["sha256"] = sha((observations / "test.npz").read_bytes())
    sensitivity = root / "evidence/sensitivity"
    sensitivity.mkdir()
    save_json(sensitivity / "training-strata.json", {
        "schema_version": 1, "protocol_sha256": protocol_hash,
        "arrays": {s: identities[s] for s in ("TRAIN", "TEST")},
        "input_sha256": {"PROTOCOL.json": protocol_hash}, "bounded_fixture": True})
    for directory, name, original in (
            (observations, "test.npz", "artifacts/inventory/test.npz"),
            (sensitivity, "training-strata.json", "artifacts/training-strata-01.json")):
        payload_hash = sha((directory / name).read_bytes())
        save_json(directory / "MANIFEST.json", {
            "schema_version": 1, "source_protocol_sha256": protocol_hash,
            "records": {name: {"source_record": original, "source_sha256": payload_hash}},
            "files": {name: payload_hash}, "scope": "Bounded public fixture.",
            "transformation": "None.", "validation": "Byte identity only."})
    real = root / "evidence/real"
    real.mkdir()
    for name in packager.REAL_FILES - {"MANIFEST.json"}:
        save_json(real / name, {"schema_version": 1, "protocol_sha256": protocol_hash,
                               "bounded_fixture": True})
    save_json(real / "protocol-summary.json", {
        "schema_version": 1, "source_protocol_sha256": protocol_hash, "arrays": identities})
    save_json(real / "MANIFEST.json", {"schema_version": 1, "output_sha256": {
        n: sha((real / n).read_bytes()) for n in packager.REAL_FILES - {"MANIFEST.json"}}})
    data = root / "evidence/data"
    data.mkdir()
    for name in packager.DATA_RECORDS:
        save_json(data / name, {"bounded_fixture": True})
    save_json(data / "MANIFEST.json", {"schema_version": 1,
        "files": {n: sha((data / n).read_bytes()) for n in packager.DATA_RECORDS},
        "records": {n: {"source_record": original, "source_sha256":
            sha((data / n).read_bytes()) if n == "inventory.json" else sha(original.encode())}
            for n, original in packager.DATA_RECORDS.items()}})
    figure(root)
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "assert sys.argv[1] == '--paper-dir' and len(sys.argv) == 3\n"
        "Path(__file__).with_suffix('.called.json').write_text(json.dumps(\n"
        "    {'argv': sys.argv, 'executable': sys.executable}))\n"
        "raise SystemExit(0)\n"
    )
    monkeypatch.setattr(packager, "VERIFIER", verifier)
    return root


def deadline_fixture(paper, *, separate=False, fixed=False, omit=()):
    """Native, bounded exports with the producer's metadata and no execution files."""
    payloads = {}

    def record(name, value):
        payloads[name] = json.dumps(value, allow_nan=False).encode()

    protocol = json.loads((paper / "evidence/real/protocol-summary.json").read_text())
    protocol_hash, arrays = protocol["source_protocol_sha256"], protocol["arrays"]
    producers = {"analysis/prigm_full_flank_gaussian.py": sha(b"Gaussian producer")}
    core = {"prigm/artifacts.py": sha(b"frozen core")}
    noise = {"stream": "full-flank-gaussian-v1", "seed": 13, "n_draws": 39,
             "key": "window and draw keyed", "hash": "SHA256", "generator": "PCG64",
             "coupling": "shared candidate noise"}
    record("gaussian/EXTENSION.json", {
        "schema_version": 1, "study": "full_flank_linear_gaussian_extension",
        "source_protocol_sha256": protocol_hash, "arrays": arrays,
        "producer_source_hashes": producers, "core_source_hashes": core,
        "numpy_version": np.__version__, "noise": noise, "scope": "Bounded synthetic fixture.",
        "model": {"lambdas": [0.001, 0.1, 1.0], "rhos": [0.0, 0.5, 1.0]},
        "interval": "39 draws; inclusive order statistics 2 and 38"})
    buffer = io.BytesIO()
    np.savez(buffer, n_train=np.array(2, dtype=np.int64), x_mean=np.zeros(216), y_mean=np.zeros(60),
             cxx=np.eye(216), cxy=np.zeros((216, 60)), cyy=np.eye(60),
             weights=np.zeros((3, 216, 60)), intercepts=np.zeros((3, 60)),
             residual_covariances=np.repeat(np.eye(60)[None], 3, axis=0))
    payloads["gaussian/FIT.npz"] = buffer.getvalue()
    fit_meta = {"schema_version": 1,
                "extension_protocol_sha256": sha(payloads["gaussian/EXTENSION.json"]),
                "fit_payload_sha256": sha(payloads["gaussian/FIT.npz"])}
    record("gaussian/FIT.json", {**fit_meta, "training_arrays": arrays["TRAIN"],
                               "fit_payload": "FIT.npz", "n_mean_fits": 3, "n_covariance_configurations": 9})
    ids = ["2018-01-01", "2018-02-01"]
    candidates = [{"lambda": lam, "rho": rho, "scores": {"raw": [1., 2.], "corrected": [1., 2.]}}
                  for lam in (0.001, 0.1, 1.) for rho in (0., 0.5, 1.)]
    record("gaussian/VALIDATION.json", {**fit_meta, "arrays": arrays["VAL"], "ids": ids,
                                      "months": [201801, 201802], "candidates": candidates})
    selected = {"raw": {"lambda": 0.1, "rho": 0, "month_balanced_primary_es": 1.5},
                "corrected": {"lambda": 1 if separate else 0.1, "rho": 1 if separate else 0,
                              "month_balanced_primary_es": 1.5}}
    names = ["lambda_0.1_rho_0"] + (["lambda_1_rho_1"] if separate else [])
    record("gaussian/SELECTION.json", {**fit_meta, "selected": selected, "selected_union": names,
                                     "validation_sha256": sha(payloads["gaussian/VALIDATION.json"])})
    record("gaussian/FIT_SELECT.json", {
        "schema_version": 1, "phase": "FIT_SELECTED", "source_protocol_sha256": protocol_hash,
        "files": {name: sha(payloads["gaussian/" + name]) for name in
                  ("EXTENSION.json", "FIT.npz", "FIT.json", "VALIDATION.json", "SELECTION.json")}})
    test_meta = {**fit_meta, "source_protocol_sha256": protocol_hash,
                 "fit_select_sha256": sha(payloads["gaussian/FIT_SELECT.json"]),
                 "selection_sha256": sha(payloads["gaussian/SELECTION.json"]), "arrays": arrays["TEST"]}
    record("gaussian/test/TEST_STARTED.json", {**test_meta, "phase": "TEST_STARTED",
                                             "previous_test_exposure": True, "selected_union": names})
    results, commits = {}, []
    for name in names:
        directory = f"gaussian/test/{name}/gen_00000000/"
        buffer = io.BytesIO()
        np.savez(buffer, ensemble=np.zeros((2, 39, 6, 48), dtype=np.float64), ids=np.array(ids))
        payloads[directory + "ensemble.npz"] = buffer.getvalue()
        lam, rho = (0.1, 0) if name == names[0] else (1, 1)
        selected_for = [k for k, row in selected.items() if (row["lambda"], row["rho"]) == (lam, rho)]
        record(directory + "COMMIT.json", {
            "schema_version": 2, "generation": 0, "files": {"ensemble.npz": sha(buffer.getvalue())},
            "protocol_sha256": fit_meta["extension_protocol_sha256"],
            "source_protocol_sha256": protocol_hash, "producer_source_hashes": producers,
            "core_source_hashes": core, "fit_select_sha256": test_meta["fit_select_sha256"],
            "selection_sha256": test_meta["selection_sha256"], "procedure": "gaussian",
            "control": {"family": "full_flank_linear_gaussian", "lambda": lam, "rho": rho,
                        "jitter": 1e-6, "joint_gaussian_channels": 5},
            "n_generated": 6, "state_convention": "physical_control_standardized_float64",
            "noise_convention": noise["stream"], "noise": noise, "arrays": arrays["TEST"],
            "training_arrays": arrays["TRAIN"], "control_fit_sha256": fit_meta["fit_payload_sha256"],
            "scope": "Bounded synthetic fixture.", "selected_for": selected_for, "complete_case": True,
            "ids_sha256": sha(json.dumps(ids, separators=(",", ":")).encode())})
        commits.append((directory + "COMMIT.json").removeprefix("gaussian/"))
        metrics = {"summary": {"primary_es": 1.5}, "per_window": [{"primary_es": 1.}, {"primary_es": 2.}]}
        results[name] = {"lambda": lam, "rho": rho, "selected_for": selected_for,
                         "ensemble": f"test/{name}", "ensemble_commit_sha256": sha(payloads[directory + "COMMIT.json"]),
                         "raw": metrics, "corrected": metrics, "correction_displacement": metrics}
    record("gaussian/test/SCORES.json", {**test_meta, "scope": "Bounded synthetic fixture.",
        "ids": ids, "months": [201801, 201802], "n": 2, "n_draws": 39, "complete_case": True,
        "selected": selected, "interval": "39 draws; inclusive order statistics 2 and 38", "candidates": results})
    record("gaussian/test/COMPLETED.json", {
        "schema_version": 1, "phase": "TEST_COMPLETED", "fit_select_sha256": test_meta["fit_select_sha256"],
        "n_windows": 2, "n_draws": 39, "selected_union": names,
        "files": {name: sha(payloads["gaussian/" + name]) for name in
                  ["test/TEST_STARTED.json", "test/SCORES.json", *commits]}})
    record("operator-mechanism.json", {"schema_version": 1, "source_protocol_sha256": protocol_hash,
                                       "bounded_fixture": True})
    if fixed:
        record("fixed-final.json", {"schema_version": 1, "original": "original-study/SUMMARY.json",
                                    "extension": "extension-study/SUMMARY.json", "bounded_fixture": True})
    originals = ["operator-mechanism.json", "gaussian/FIT_SELECT.json"] + (["fixed-final.json"] if fixed else [])
    record("MANIFEST.json", {"schema_version": 1, "source_protocol_sha256": protocol_hash,
        "files": {name: sha(payload) for name, payload in payloads.items()},
        "original_sha256": {name: sha(payloads[name]) for name in originals},
        "producers": {**producers, "analysis/prigm_operator_mechanism.py": sha(b"operator producer")},
        "transformations": {k: v for k, v in packager.DEADLINE_TRANSFORMATIONS.items()
                            if k != "fixed-final.json" or fixed}})
    for name, payload in payloads.items():
        if name not in omit:
            path = paper / "evidence/deadline" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return payloads


def rehash_deadline(paper):
    """Update only the public manifest so inner binding checks remain testable."""
    directory = paper / "evidence/deadline"
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    manifest["files"] = {name: sha((directory / name).read_bytes()) for name in manifest["files"]}
    for name in ("operator-mechanism.json", "gaussian/FIT_SELECT.json"):
        manifest["original_sha256"][name] = manifest["files"][name]
    save_json(directory / "MANIFEST.json", manifest)


@pytest.fixture
def calibration(paper, monkeypatch):
    deadline_fixture(paper)
    directory = paper / "evidence/followup/calibration"
    directory.mkdir(parents=True)
    producer = sha(b"bounded calibration producer")
    methods = {}
    arrays = {"ids": np.array(["2018-01-01", "2018-02-01"]), "months": np.array([201801, 201802])}
    for name in ("D", "I-S", "O-S", "G5", "gaussian", "full_flank_gaussian"):
        seeds = [None] if "gaussian" in name else [7, 13, 42]
        mean = np.tile([.25, .5, .25], (5, 1))
        methods[name] = {"fixed_model_seeds": seeds, "mean": mean.tolist(),
                         "model_means": np.tile(mean, (len(seeds), 1, 1)).tolist(),
                         "year_95": np.tile(mean, (2, 1, 1)).tolist()}
        arrays[name] = np.tile(mean, (len(seeds), 2, 1, 1))
    np.savez(directory / "per_window.npz", **arrays)
    summary = {"schema_version": 1, "channels": ["mu_x", "mu_y", "mu_z", "m", "v_s"],
               "per_window_sha256": sha((directory / "per_window.npz").read_bytes()),
               "bootstrap": {"replicates": 10000, "seed": 7003, "schemes": ["year", "twoyear"]},
               "source_sha256": {"analysis/prigm_tail_calibration.py": producer},
               "input_sha256": {"original-study/artifacts/inventory/test.npz":
                                 sha((paper / "evidence/observations/test.npz").read_bytes())},
               "methods": methods}
    save_json(directory / "summary.json", summary)
    save_json(directory / "PROTOCOL.json", {"schema_version": 1, "new_analysis_sha256": producer,
              "input_manifests": {group + "/MANIFEST.json":
                                  sha((paper / f"evidence/{group}/MANIFEST.json").read_bytes())
                                  for group in ("real", "deadline")}})
    save_json(directory / "MANIFEST.json", {
        "schema_version": 1,
        "files": {n: sha((directory / n).read_bytes())
                  for n in ("summary.json", "PROTOCOL.json", "per_window.npz")},
        "original_summary_sha256": sha(b"private summary"), "producer_sha256": producer,
        "protocol_sha256": sha((directory / "PROTOCOL.json").read_bytes()),
        "source": {"path": "export_prigm_calibration.py", "sha256": sha(b"exporter")},
        "transformation": packager.CALIBRATION_TRANSFORMATION})
    figure(paper, "calibration", "evidence/followup/calibration/summary.json")
    path = paper / "figures/calibration/fig_calibration.manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update({
        "protocol_sha256": sha(b"real protocol"), "channels": summary["channels"],
        "outcomes": ["below", "inside", "above"],
        "method_order": ["D", "O-S", "G5", "full_flank_gaussian"],
        "interval_order_statistics": [2, 38], "pointwise_intervals_plotted": False,
        "calibration_provenance": {k: summary[k] for k in ("bootstrap", "per_window_sha256")},
        "decomposition": [{"seed": seed, "delta_target": -.01,
                           "negative_delta_dispersion": -.09, "delta_energy": -.1}
                          for seed in (7, 13, 42, "mean")],
        "tails": {method: {"fixed_model_seeds": methods[method]["fixed_model_seeds"],
                            **{target: (100 * np.asarray(methods[method][field])).tolist()
                               for target, field in (("mean_percent", "mean"),
                                  ("model_means_percent", "model_means"), ("year_95_percent", "year_95"))}}
                  for method in ("D", "O-S", "G5", "full_flank_gaussian")},
        "inputs": {"operator": {"path": "evidence/deadline/operator-mechanism.json",
                                "sha256": sha((paper / "evidence/deadline/operator-mechanism.json").read_bytes())},
                   "calibration": {"path": "evidence/followup/calibration/summary.json",
                                   "sha256": sha((directory / "summary.json").read_bytes())}}})
    save_json(path, manifest)
    source = paper / "main.tex"
    source.write_text(source.read_text().replace(r"\end{document}",
                      r"\includegraphics{figures/calibration/fig_calibration.pdf}\end{document}"))
    helpers = {}
    for name in packager.FOLLOWUP_VERIFIERS:
        helper = paper.parent / Path(name).name
        helper.write_text('"""Bounded verifier support fixture."""\n')
        helpers[name] = helper
    monkeypatch.setattr(packager, "FOLLOWUP_VERIFIERS", helpers)
    checkpoint_fixture(paper)
    return paper


def checkpoint_fixture(paper):
    directory = paper / "evidence/followup/checkpoints"
    directory.mkdir(parents=True)
    originals = {n: sha(("private " + n).encode()) for n in packager.CHECKPOINT_FILES - {"MANIFEST.json"}}
    common = {"protocol_sha256": originals["PROTOCOL.json"], "scope_sha256": originals["SCOPE.json"]}
    curves = {f"{arm}_s{seed}": [4., 3., 2., 3., 4.] for arm in ("D", "O-S") for seed in (7, 13, 42)}
    policies = {name: {"P25": 16252, "P50": 16252, "F": 29636} for name in curves}
    save_json(directory / "PROTOCOL.json", {"scope_sha256": originals["SCOPE.json"],
              "environment": {"executable": "environment-python", "hostname": "execution-host"}})
    save_json(directory / "SCOPE.json", {"bounded_fixture": True})
    save_json(directory / "SELECTION.json", {**common, "policies": policies})
    save_json(directory / "SAVED_VAL25.json", {**common, "curves": {n: {"means": v} for n, v in curves.items()}})
    save_json(directory / "SUMMARY.json", {**common, "selection_sha256": originals["SELECTION.json"],
              "input_sha256": {"checkpoint-study/SAVED_VAL25.json": originals["SAVED_VAL25.json"]},
              "policies": policies, "curves": {"VAL": curves, "TEST": curves}})
    figure(paper, "checkpoints", "evidence/followup/checkpoints/SUMMARY.json")
    source = {"path": "fig_checkpoints.py", "sha256": sha((paper / "fig_checkpoints.py").read_bytes())}
    save_json(directory / "MANIFEST.json", {
        "schema_version": 1, "analysis": "checkpoint_followup_public_export",
        "source_protocol_sha256": originals["PROTOCOL.json"], "source_scope_sha256": originals["SCOPE.json"],
        "source": source, "original_sha256": originals,
        "files": {n: sha((directory / n).read_bytes()) for n in originals},
        "transformations": {"root_replacements": [
            {"original_root_sha256": sha(name.encode()), "public_root": name}
            for name in ("checkpoint-study", "original-study", "fixed-final-study")],
            "field_replacements": [{"file": "PROTOCOL.json", "field": "environment." + field,
                                    "original_value_sha256": sha(field.encode()), "public_value": value}
                                   for field, value in (("executable", "environment-python"),
                                                        ("hostname", "execution-host"))]}})
    path = paper / "figures/checkpoints/fig_checkpoints.manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update({"protocol_sha256": originals["PROTOCOL.json"], "policies": policies,
                     "curves": {"VAL25": curves, "VAL50": curves, "TEST50": curves},
                     "inputs": {n: {"path": "evidence/followup/checkpoints/" + n,
                                    "sha256": sha((directory / n).read_bytes())}
                                for n in packager.CHECKPOINT_FILES}})
    save_json(path, manifest)
    source = paper / "main.tex"
    source.write_text(source.read_text().replace(r"\end{document}",
                      r"\includegraphics{figures/checkpoints/fig_checkpoints.pdf}\end{document}"))


def run(paper, output):
    return packager.main(["--paper-dir", str(paper), "--output", str(output)])


def assert_nothing(output):
    assert not os.path.lexists(output)
    assert not os.path.lexists(str(output) + ".manifest.json")
    assert not list(output.parent.glob(".prigm-package-*"))


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_calibration_roundtrip_and_verifier_receipt(calibration, tmp_path, suffix):
    output = tmp_path / ("calibration" + suffix)
    assert run(calibration, output) == 0
    members = read_archive(output)
    expected = {"evidence/followup/calibration/" + n for n in packager.CALIBRATION_FILES}
    expected |= {"fig_calibration.py", *("figures/calibration/fig_calibration." + ext
                                        for ext in ("pdf", "png", "manifest.json"))}
    expected |= {"evidence/followup/checkpoints/" + n for n in packager.CHECKPOINT_FILES}
    expected |= {"fig_checkpoints.py", *("figures/checkpoints/fig_checkpoints." + ext
                                        for ext in ("pdf", "png", "manifest.json"))}
    assert expected <= members.keys()
    for name in expected:
        assert members[name] == (calibration / name).read_bytes()
    receipt = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert receipt["verification"]["supporting_sources"] == {
        name: sha(path.read_bytes()) for name, path in packager.FOLLOWUP_VERIFIERS.items()}
    assert packager.VERIFIER.with_suffix(".called.json").is_file()


@pytest.mark.parametrize("damage", ["missing", "extra", "unknown_group", "empty", "symlink"])
def test_followup_inventory_is_closed(calibration, tmp_path, damage):
    directory = calibration / "evidence/followup/calibration"
    if damage == "missing":
        (directory / "PROTOCOL.json").unlink()
    elif damage == "extra":
        (directory / "extra.json").write_text("{}")
    elif damage == "unknown_group":
        (directory.parent / "unknown").mkdir()
    elif damage == "empty":
        for path in directory.iterdir():
            path.unlink()
    else:
        path = directory / "per_window.npz"
        payload = tmp_path / "array.npz"
        path.replace(payload)
        path.symlink_to(payload)
    output = tmp_path / "bad-followup.zip"
    assert run(calibration, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("filename,keys,value,message", [
    ("MANIFEST.json", ["schema_version"], True, "inventory/transformation"),
    ("MANIFEST.json", ["files", "unknown.json"], "0" * 64, "inventory/transformation"),
    ("MANIFEST.json", ["files", "summary.json"], "0" * 64, "inventory/transformation"),
    ("MANIFEST.json", ["transformation"], "Changed numerical values.", "inventory/transformation"),
    ("MANIFEST.json", ["original_summary_sha256"], "bad", "SHA256"),
    ("MANIFEST.json", ["source", "path"], "other.py", "exporter identity"),
    ("MANIFEST.json", ["source", "sha256"], "bad", "SHA256"),
    ("MANIFEST.json", ["protocol_sha256"], "0" * 64, "protocol/producer/input"),
    ("MANIFEST.json", ["producer_sha256"], "0" * 64, "protocol/producer/input"),
    ("PROTOCOL.json", ["new_analysis_sha256"], "0" * 64, "protocol/producer/input"),
    ("PROTOCOL.json", ["input_manifests", "real/MANIFEST.json"], "0" * 64, "protocol/producer/input"),
    ("summary.json", ["per_window_sha256"], "0" * 64, "protocol/producer/input"),
    ("summary.json", ["source_sha256", "analysis/prigm_tail_calibration.py"], "0" * 64, "protocol/producer/input"),
    ("summary.json", ["input_sha256", "../escape.json"], "0" * 64, "unsafe"),
])
def test_calibration_rehashed_records_keep_inner_bindings(calibration, tmp_path, capsys,
                                                         filename, keys, value, message):
    directory = calibration / "evidence/followup/calibration"
    path = directory / filename
    record = json.loads(path.read_text())
    target = record
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    save_json(path, record)
    if filename != "MANIFEST.json":
        manifest_path = directory / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][filename] = sha(path.read_bytes())
        if filename == "PROTOCOL.json":
            manifest["protocol_sha256"] = manifest["files"][filename]
        save_json(manifest_path, manifest)
    output = tmp_path / "bad-calibration.zip"
    assert run(calibration, output) == 1
    assert message in capsys.readouterr().err
    assert_nothing(output)


@pytest.mark.parametrize("keys,value,message", [
    (["protocol_sha256"], "0" * 64, "schema/provenance"),
    (["channels"], ["wrong"], "schema/provenance"),
    (["pointwise_intervals_plotted"], True, "schema/provenance"),
    (["pointwise_intervals_plotted"], 0, "schema/provenance"),
    (["calibration_provenance", "per_window_sha256"], "0" * 64, "schema/provenance"),
    (["interval_order_statistics"], [1, 39], "schema/provenance"),
    (["inputs", "calibration", "path"], "MANIFEST.json", "missing/ambiguous"),
    (["inputs", "calibration", "sha256"], "0" * 64, "missing/ambiguous"),
    (["inputs", "calibration", "path"], "../summary.json", "unsafe"),
    (["decomposition"], [], "decomposition inventory"),
    (["decomposition", 0, "delta_energy"], "-0.1", "finite numeric"),
    (["tails", "D", "mean_percent"], [[0, 0, 0]] * 5, "plotted tail values"),
    (["tails", "D", "model_means_percent"], [[[25, 50, 25]] * 5], "plotted tail values"),
    (["tails", "D", "year_95_percent"], [[[25, 50, 26]] * 5] * 2, "plotted tail values"),
    (["tails", "D", "fixed_model_seeds"], [42, 13, 7], "model order"),
])
def test_calibration_figure_numeric_manifest(calibration, tmp_path, capsys, keys, value, message):
    path = calibration / "figures/calibration/fig_calibration.manifest.json"
    record = json.loads(path.read_text())
    target = record
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    save_json(path, record)
    output = tmp_path / "bad-calibration-figure.zip"
    assert run(calibration, output) == 1
    assert message in capsys.readouterr().err
    assert_nothing(output)


@pytest.mark.parametrize("label", ["fig:calibration", "fig:checkpoints", "app:checkpoints",
                                  r"\includegraphics{figures/calibration/fig_calibration.pdf}"])
def test_followup_claims_require_the_new_evidence_group(paper, tmp_path, label):
    source = paper / "main.tex"
    source.write_text(source.read_text() + label)
    output = tmp_path / "missing-followup.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


def test_calibration_evidence_requires_referenced_figure(calibration, tmp_path):
    source = calibration / "main.tex"
    source.write_text(source.read_text().replace(
        r"\includegraphics{figures/calibration/fig_calibration.pdf}", ""))
    output = tmp_path / "missing-calibration-figure.zip"
    assert run(calibration, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("change", ["_prigm_camera_ready.py", "_prigm_checkpoint_evidence.py", "_prigm_documentation.py", "summary", "new_group"])
def test_followup_changes_during_verifier_refuse_publication(calibration, tmp_path, monkeypatch, change):
    original = packager.subprocess.run

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if change.endswith(".py"):
            packager.FOLLOWUP_VERIFIERS["experiments/" + change].write_text("# changed verifier support\n")
        elif change == "summary":
            path = calibration / "evidence/followup/calibration/summary.json"
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            (calibration / "evidence/followup/new-group").mkdir()
        return result
    monkeypatch.setattr(packager.subprocess, "run", mutate)
    output = tmp_path / "changed-followup.zip"
    assert run(calibration, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("filename,keys,value,message", [
    ("MANIFEST.json", ["schema_version"], True, "inventory/source"),
    ("MANIFEST.json", ["files", "extra.json"], "0" * 64, "inventory/source"),
    ("MANIFEST.json", ["original_sha256", "SUMMARY.json"], "bad", "SHA256"),
    ("MANIFEST.json", ["source_protocol_sha256"], "0" * 64, "original/public provenance"),
    ("MANIFEST.json", ["source_scope_sha256"], "0" * 64, "original/public provenance"),
    ("MANIFEST.json", ["transformations", "root_replacements", 0, "public_root"], "unknown", "root transformations"),
    ("MANIFEST.json", ["transformations", "field_replacements", 0, "public_value"], "unknown", "environment alias"),
    ("MANIFEST.json", ["transformations", "field_replacements", 0, "original_value_sha256"], "bad", "SHA256"),
    ("MANIFEST.json", ["transformations", "field_replacements", 0, "field"], "other", "field transformations"),
    ("PROTOCOL.json", ["scope_sha256"], "0" * 64, "original/public provenance"),
    ("PROTOCOL.json", ["environment", "hostname"], "unknown", "environment alias"),
    ("SELECTION.json", ["protocol_sha256"], "0" * 64, "original/public provenance"),
    ("SAVED_VAL25.json", ["scope_sha256"], "0" * 64, "original/public provenance"),
    ("SUMMARY.json", ["selection_sha256"], "0" * 64, "original/public provenance"),
    ("SUMMARY.json", ["input_sha256", "checkpoint-study/SAVED_VAL25.json"], "0" * 64, "original/public provenance"),
])
def test_checkpoint_public_hashes_do_not_replace_original_bindings(calibration, tmp_path, capsys,
                                                                  filename, keys, value, message):
    directory = calibration / "evidence/followup/checkpoints"
    path = directory / filename
    record = json.loads(path.read_text())
    target = record
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    save_json(path, record)
    if filename != "MANIFEST.json":
        manifest_path = directory / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][filename] = sha(path.read_bytes())
        save_json(manifest_path, manifest)
    output = tmp_path / "bad-checkpoint.zip"
    assert run(calibration, output) == 1
    assert message in capsys.readouterr().err
    assert_nothing(output)


@pytest.mark.parametrize("filename", sorted(packager.CHECKPOINT_FILES))
def test_checkpoint_inventory_requires_every_payload(calibration, tmp_path, filename):
    (calibration / "evidence/followup/checkpoints" / filename).unlink()
    output = tmp_path / "incomplete-checkpoint.zip"
    assert run(calibration, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("damage", [None, "missing", "incorrect"])
def test_checkpoint_package_separates_exporter_from_renderer(calibration, damage):
    provenance_name = "evidence/followup/checkpoints/MANIFEST.json"
    original_bytes = (calibration / provenance_name).read_bytes()
    exporter = json.loads(original_bytes)["source"]
    source = calibration / "fig_checkpoints.py"
    source.write_bytes(source.read_bytes() + b"\n# Presentation-only renderer revision.\n")
    renderer = {"path": source.name, "sha256": sha(source.read_bytes())}
    assert renderer != exporter
    figure_name = "figures/checkpoints/fig_checkpoints.manifest.json"
    manifest = json.loads((calibration / figure_name).read_text())
    manifest["source"] = renderer
    manifest["evidence_source"] = exporter
    if damage == "missing":
        del manifest["evidence_source"]
    elif damage == "incorrect":
        manifest["evidence_source"] = renderer.copy()
    save_json(calibration / figure_name, manifest)
    if damage:
        with pytest.raises(ValueError, match="checkpoint figure numeric/provenance mismatch"):
            packager.collect(calibration)
    else:
        files, _, _ = packager.collect(calibration)
        assert files[source.name] == source.read_bytes()
        assert files[provenance_name] == original_bytes
        packaged = json.loads(files[figure_name])
        assert packaged["source"] == renderer
        assert packaged["evidence_source"] == exporter
    assert (calibration / provenance_name).read_bytes() == original_bytes


@pytest.mark.parametrize("damage", ["curve", "policy", "source", "ambiguous_manifest", "extra_output",
                                     "missing_model", "string_curve", "short_curve"])
def test_checkpoint_figure_is_bound_to_public_numeric_records(calibration, tmp_path, damage):
    path = calibration / "figures/checkpoints/fig_checkpoints.manifest.json"
    record = json.loads(path.read_text())
    if damage == "curve":
        record["curves"]["VAL25"]["D_s7"][0] += .01
    elif damage == "policy":
        record["policies"]["D_s7"]["P25"] = 2868
    elif damage == "source":
        record["source"]["sha256"] = "0" * 64
    elif damage == "ambiguous_manifest":
        record["inputs"]["MANIFEST.json"]["path"] = "MANIFEST.json"
    elif damage == "extra_output":
        record["output_sha256"]["checkpoint-summary.json"] = "0" * 64
    elif damage == "missing_model":
        del record["curves"]["VAL25"]["D_s7"]
    elif damage == "string_curve":
        record["curves"]["VAL25"]["D_s7"][0] = "4.0"
    else:
        record["curves"]["VAL25"]["D_s7"] = [4., 3.]
    save_json(path, record)
    output = tmp_path / "bad-checkpoint-figure.zip"
    assert run(calibration, output) == 1
    assert_nothing(output)


def test_checkpoint_curves_allow_reduction_roundoff(calibration, tmp_path):
    path = calibration / "figures/checkpoints/fig_checkpoints.manifest.json"
    record = json.loads(path.read_text())
    record["curves"]["VAL25"]["D_s7"][0] = float(np.nextafter(4., np.inf))
    save_json(path, record)
    assert run(calibration, tmp_path / "roundoff.zip") == 0


def read_archive(output):
    if output.suffix == ".zip":
        with zipfile.ZipFile(output) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            for member in archive.infolist():
                assert stat.S_ISREG(member.external_attr >> 16)
            return {name: archive.read(name) for name in archive.namelist()}
    with tarfile.open(output, "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members})
        assert all(m.isfile() and not m.uname and not m.gname and m.uid == m.gid == 0 for m in members)
        return {member.name: archive.extractfile(member).read() for member in members}


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_contents_roundtrip_receipt_and_verifier(paper, tmp_path, suffix):
    output = tmp_path / ("anonymous" + suffix)
    assert run(paper, output) == 0
    call = json.loads(packager.VERIFIER.with_suffix(".called.json").read_text())
    assert call == {"argv": [str(packager.VERIFIER), "--paper-dir", str(paper)],
                    "executable": sys.executable}
    members = read_archive(output)
    assert set(members) == {
        "LICENSE", "main.tex", "references.bib", "neurips_2025.sty", "fig_hmm.py",
        "figures/hmm/fig_hmm.pdf", "figures/hmm/fig_hmm.png",
        "figures/hmm/fig_hmm.manifest.json",
        "evidence/toy/oracle-stationary-01.json", "evidence/toy/hmm-stationary-01.json",
        "evidence/toy/toy.npz", "evidence/toy/learned/learned.json",
        "evidence/toy/learned/reference-calibration.json", "evidence/toy/learned/MANIFEST.json",
        "evidence/real/comparison.json", "evidence/real/diagnostics.json", "evidence/real/scores.json",
        "evidence/real/protocol-summary.json", "evidence/real/gradients.json", "evidence/real/MANIFEST.json",
        "evidence/data/quality.json", "evidence/data/q15.json", "evidence/data/inventory.json",
        "evidence/data/MANIFEST.json",
        "evidence/observations/test.npz", "evidence/observations/MANIFEST.json",
        "evidence/sensitivity/training-strata.json", "evidence/sensitivity/MANIFEST.json",
    }
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert manifest["sha256"] == sha(output.read_bytes())
    assert manifest["included"] == {
        name: {"sha256": sha(payload), "bytes": len(payload)} for name, payload in members.items()}
    assert manifest["verification"]["sha256"] == sha(packager.VERIFIER.read_bytes())
    assert manifest["verification"]["returncode"] == 0
    assert manifest["excluded"] and "Not performed" in manifest["compilation"]
    assert members["neurips_2025.sty"] == (paper / "neurips_2025.sty").read_bytes()
    for name, payload in members.items():
        assert payload == ((REPO if name == "LICENSE" else paper) / name).read_bytes()
    assert not list(tmp_path.glob(".prigm-package-*"))


@pytest.mark.parametrize("text", [
    "other@example.edu", "other@example.com", "ALISON", "/Users/private/study",
    r"C:\Users\private\study", "/home/private/study", "/tmp/private/study",
])
def test_camera_ready_only_exempts_declared_author(text):
    with pytest.raises(ValueError, match="author/path leak"):
        packager.scan_text(text, "main.tex", camera_ready=True)


def camera_ready_source(tex):
    return (tex.replace("dblblindworkshop", "final,sglblindworkshop")
            .replace(packager.NOTICE_ANONYMOUS, packager.NOTICE_CAMERA_READY)
            .replace("pdfauthor={}", "pdfauthor={Kevin Lee}")
            .replace(r"\author{Anonymous}", r"\author{Kevin Lee \orcidlink{0009-0004-0388-9260}"
                     r"\\University of California at Los Angeles"
                     r"\\NASA Jet Propulsion Laboratory"
                     r"\\kevinlee69720@g.ucla.edu}")
            .replace(r"\usepackage{graphicx,hyperref}", r"\usepackage{graphicx,hyperref}\usepackage{orcidlink}"))


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_camera_ready_bundle_preserves_author_and_license(paper, tmp_path, suffix):
    source = paper / "main.tex"
    source.write_text(camera_ready_source(source.read_text()))
    output = tmp_path / ("camera-ready" + suffix)
    assert run(paper, output) == 0
    members = read_archive(output)
    assert members["main.tex"] == source.read_bytes()
    assert members["LICENSE"] == (REPO / "LICENSE").read_bytes()
    with pytest.raises(ValueError, match="author/path leak"):
        packager.scan_text(packager.EMAIL_CAMERA_READY, "evidence/record.json", camera_ready=True)


def test_license_changes_during_verification_refuse_publication(paper, tmp_path, monkeypatch):
    license_path = tmp_path / "LICENSE"
    license_path.write_bytes((REPO / "LICENSE").read_bytes())
    monkeypatch.setattr(packager, "LICENSE", license_path)
    original = packager.subprocess.run

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        license_path.write_text("Changed notice")
        return result

    monkeypatch.setattr(packager.subprocess, "run", mutate)
    output = tmp_path / "changed-license.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


def test_cli_default_and_required_output(paper, tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "DEFAULT_PAPER", paper)
    assert packager.main(["--output", str(tmp_path / "default.zip")]) == 0
    with pytest.raises(SystemExit) as exc:
        packager.main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_verifier_refusal_propagates_without_publication(paper, tmp_path, suffix):
    packager.VERIFIER.write_text(packager.VERIFIER.read_text().replace("SystemExit(0)", "SystemExit(23)"))
    output = tmp_path / ("refused" + suffix)
    assert run(paper, output) == 23
    assert packager.VERIFIER.with_suffix(".called.json").exists()
    assert_nothing(output)


def test_missing_verifier_is_not_a_bypass(paper, tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "VERIFIER", tmp_path / "not-implemented.py")
    output = tmp_path / "missing-gate.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("existing", ["archive", "sidecar"])
@pytest.mark.parametrize("symlink", [False, True])
@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_fresh_archive_and_full_suffix_sidecar_refusal(paper, tmp_path, existing, symlink, suffix):
    output = tmp_path / ("occupied" + suffix)
    sidecar = Path(str(output) + ".manifest.json")
    target = output if existing == "archive" else sidecar
    if symlink:
        target.symlink_to(tmp_path / "absent-target")
    else:
        target.write_bytes(b"unfamiliar bytes")
    assert run(paper, output) == 1
    assert target.is_symlink() if symlink else target.read_bytes() == b"unfamiliar bytes"
    other = sidecar if existing == "archive" else output
    assert not other.exists()
    assert not packager.VERIFIER.with_suffix(".called.json").exists()
    assert not list(tmp_path.glob(".prigm-package-*"))


@pytest.mark.parametrize("name", ["figures/hmm/fig_hmm.pdf", "figures/hmm/fig_hmm.png",
                                  "figures/hmm/fig_hmm.manifest.json", "fig_hmm.py", "references.bib"])
def test_missing_dependency(paper, tmp_path, name):
    (paper / name).unlink()
    output = tmp_path / "missing.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("name", ["references.bib", "fig_hmm.py", "evidence/toy", "figures/hmm"])
def test_outside_symlink(paper, tmp_path, name):
    original = paper / name
    outside = tmp_path / "outside"
    original.rename(outside)
    original.symlink_to(outside, target_is_directory=outside.is_dir())
    output = tmp_path / "symlink.zip"
    assert run(paper, output) == 1
    assert outside.exists() and original.is_symlink()
    assert_nothing(output)


@pytest.mark.parametrize("reference", [
    r"\input{../outside}", r"\input{/tmp/outside.tex}", r"\input{chapters/../../outside.tex}",
    r"\input{\chosenfile}", r"\input outside.tex", r"\input{model.pt}",
    r"\includegraphics{figures/hmm/fig_hmm}", r"\includegraphics{fig_hmm.pdf}",
    r"\graphicspath{{../outside/}}", r"\InputIfFileExists{outside.tex}{}{}",
    r"\usepackage{../../outside}", r"\def\file{outside}\input{\file}",
    r"\bibliography{references,other}", r"\bibliographystyle{plainnat}", r"\usepackage[final]{neurips_2025}",
    r"\PassOptionsToPackage{final}{neurips_2025}", r"\makeatletter\@neuripsfinaltrue\makeatother",
])
def test_traversal_and_ambiguous_tex_references(paper, tmp_path, reference):
    source = paper / "main.tex"
    source.write_text(source.read_text().replace(r"\end{document}", reference + "\n" + r"\end{document}"))
    output = tmp_path / "unsafe.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


def test_literal_recursive_inputs_are_root_relative(paper, tmp_path):
    (paper / "sections").mkdir()
    (paper / "sections/one.tex").write_text(r"A section. \input{sections/two.tex}")
    (paper / "sections/two.tex").write_text("Another section.")
    source = paper / "main.tex"
    source.write_text(source.read_text().replace(r"\maketitle", r"\maketitle\input{sections/one}"))
    output = tmp_path / "sections.zip"
    assert run(paper, output) == 0
    assert {"sections/one.tex", "sections/two.tex"} <= read_archive(output).keys()


@pytest.mark.parametrize("leak", ["Kevin Lee", "Kevin~Lee", "Kevin_Lee", "Alison", "magnaprog", "skykev1928",
                                  "/home/private/file", "/tmp/run/file", "person@colorado.edu",
                                  "person@ucla.edu", "person@institute.ac.uk", "person@jpl.nasa.gov"])
def test_concrete_source_leaks(paper, tmp_path, leak):
    source = paper / "main.tex"
    source.write_text(source.read_text() + "\n% " + leak)
    output = tmp_path / "leak.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("kind", ["json", "escaped-json", "unicode-npz", "bytes-npz", "structured-npz", "generator", "bib"])
def test_evidence_and_companion_text_leaks(paper, tmp_path, kind):
    if kind == "json":
        save_json(paper / "evidence/toy/oracle-stationary-01.json", {"path": "/home/private/run"})
    elif kind == "escaped-json":
        (paper / "evidence/toy/oracle-stationary-01.json").write_text(r'{"path":"\u002fhome\u002fprivate"}')
    elif kind.endswith("npz"):
        if kind == "unicode-npz":
            array = np.array(["magnaprog"])
        elif kind == "bytes-npz":
            array = np.array([b"/tmp/private"])
        else:
            array = np.array([("Alison",)], dtype=[("label", "U12")])
        np.savez(paper / "evidence/toy/toy.npz", strings=array)
    else:
        target = paper / ("fig_hmm.py" if kind == "generator" else "references.bib")
        target.write_text(target.read_text() + "\n# Kevin Lee\n")
    output = tmp_path / "leak-evidence.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("placeholder", ["TODO", "TBD", "PLACEHOLDER", "Reanalysis in progress; not ready for submission."])
def test_placeholders_refused_even_with_successful_stub(paper, tmp_path, placeholder):
    source = paper / "main.tex"
    source.write_text(source.read_text() + "\n" + placeholder)
    output = tmp_path / "unfinished.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


def test_nonanonymous_author_and_modified_vendor_refused(paper, tmp_path):
    source = paper / "main.tex"
    source.write_text(source.read_text().replace("{Anonymous}", "{Some Author}"))
    output = tmp_path / "named.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    source.write_text(source.read_text().replace("{Some Author}", "{Anonymous}"))
    style = paper / "neurips_2025.sty"
    style.write_bytes(style.read_bytes() + b"\n% changed vendor\n")
    assert run(paper, output) == 1
    assert_nothing(output)


def test_object_npz_never_unpickles(paper, tmp_path, monkeypatch):
    np.savez(paper / "evidence/toy/toy.npz", objects=np.array([{"private": "value"}], dtype=object))
    calls = []
    original_load = np.load

    def guarded_load(*args, **kwargs):
        calls.append(kwargs.get("allow_pickle"))
        assert kwargs["allow_pickle"] is False
        return original_load(*args, **kwargs)
    monkeypatch.setattr(packager.np, "load", guarded_load)
    output = tmp_path / "object.zip"
    assert run(paper, output) == 1
    assert calls and not any(calls)
    assert_nothing(output)


def test_unused_historical_and_unfamiliar_files_stay_untouched(paper, tmp_path):
    extras = ["fig_hmm.pdf", "fig_hmm.png", "fig_audit.pdf", "fig_cycle.py", "main.pdf", "main.aux",
              "main.log", "main.bbl", "neurips_2026.sty", "planning.md", "model.pt", ".env",
              "pisces-overleaf.zip", "pisces-arxiv.tar.gz", "unknown.bin"]
    for name in extras:
        (paper / name).write_bytes(b"unfamiliar private /home/ contents")
    (paper / "unused-link").symlink_to(tmp_path / "absent")
    (paper / "private-directory").mkdir()
    (paper / "private-directory/secret.txt").write_text("Kevin Lee")
    output = tmp_path / "clean.zip"
    assert run(paper, output) == 0
    assert not set(extras) & read_archive(output).keys()
    assert all((paper / name).read_bytes() == b"unfamiliar private /home/ contents" for name in extras)
    assert (paper / "unused-link").is_symlink()


@pytest.mark.parametrize("name,payload", [("evidence/toy/model.pt", b"checkpoint"),
                                         ("evidence/toy/log.txt", b"execution log"),
                                         ("evidence/toy/bad.json", b"\x00binary")])
def test_unexpected_evidence_type(paper, tmp_path, name, payload):
    (paper / name).write_bytes(payload)
    output = tmp_path / "binary.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


def test_required_real_evidence_and_optional_audit_figure(paper, tmp_path):
    figure(paper, "audit", "evidence/real/comparison.json")
    source = paper / "main.tex"
    source.write_text(source.read_text().replace(r"\end{document}",
                      r"\includegraphics{figures/audit/fig_audit.pdf}\end{document}"))
    output = tmp_path / "real.tar.gz"
    assert run(paper, output) == 0
    assert {"evidence/real/" + n for n in packager.REAL_FILES} <= read_archive(output).keys()
    assert "fig_audit.py" in read_archive(output)


def test_hmm_schema2_preserves_companion_bytes(paper):
    files, _, _ = packager.collect(paper)
    manifest_name = "figures/hmm/fig_hmm.manifest.json"
    assert json.loads(files[manifest_name])["schema_version"] == 2
    for name in (manifest_name, "fig_hmm.py", "figures/hmm/fig_hmm.pdf",
                 "figures/hmm/fig_hmm.png", "evidence/toy/hmm-stationary-01.json"):
        assert files[name] == (paper / name).read_bytes()


@pytest.mark.parametrize("schema", [0, 1, 3, True, False, None, "2", 2.0, "missing"])
def test_hmm_rejects_other_schema_versions(paper, schema):
    packager.collect(paper)
    path = paper / "figures/hmm/fig_hmm.manifest.json"
    manifest = json.loads(path.read_text())
    if schema == "missing":
        del manifest["schema_version"]
    else:
        manifest["schema_version"] = schema
    save_json(path, manifest)
    with pytest.raises((ValueError, KeyError)):
        packager.collect(paper)


@pytest.mark.parametrize("schema", [2, True, 1.0])
def test_other_figures_keep_integer_schema1(paper, schema):
    figure(paper, "audit", "evidence/real/comparison.json")
    source = paper / "main.tex"
    source.write_text(source.read_text().replace(r"\end{document}",
                      r"\includegraphics{figures/audit/fig_audit.pdf}\end{document}"))
    packager.collect(paper)
    path = paper / "figures/audit/fig_audit.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = schema
    save_json(path, manifest)
    with pytest.raises(ValueError, match="figure companion/generator SHA256 mismatch"):
        packager.collect(paper)


@pytest.mark.parametrize("change", ["pdf", "png", "generator", "input", "manifest-path"])
def test_checked_figure_companions_and_inputs(paper, tmp_path, change):
    packager.collect(paper)
    paths = {"pdf": "figures/hmm/fig_hmm.pdf", "png": "figures/hmm/fig_hmm.png",
             "generator": "fig_hmm.py", "input": "evidence/toy/hmm-stationary-01.json"}
    if change == "manifest-path":
        path = paper / "figures/hmm/fig_hmm.manifest.json"
        record = json.loads(path.read_text())
        record["inputs"]["record"]["path"] = "../outside.json"
        save_json(path, record)
    else:
        path = paper / paths[change]
        path.write_bytes(path.read_bytes() + b"\n ")
    output = tmp_path / "stale.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("when", ["verifier", "write", "readback"])
@pytest.mark.parametrize("change", ["mutate", "replace", "remove"])
def test_source_mutation_and_replacement_publish_nothing(paper, tmp_path, monkeypatch, when, change):
    source = paper / "references.bib"

    def mutate():
        if change == "mutate":
            source.write_bytes(source.read_bytes() + b"\n")
        elif change == "replace":
            replacement = paper / "replacement.bib"
            replacement.write_bytes(source.read_bytes())
            replacement.replace(source)
        else:
            source.unlink()
    if when == "verifier":
        original = packager.subprocess.run

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(packager.subprocess, "run", wrapped)
    else:
        function = "write_archive" if when == "write" else "verify_archive"
        original = getattr(packager, function)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(packager, function, wrapped)
    output = tmp_path / "mutated.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
@pytest.mark.parametrize("change", ["bytes", "missing", "extra", "duplicate", "nonregular", "truncate"])
def test_corrupt_archive_never_publishes(paper, tmp_path, monkeypatch, suffix, change):
    original = packager.write_archive

    def corrupt(path, kind, files):
        altered = dict(files)
        if change == "bytes":
            altered["main.tex"] += b"corruption"
        elif change == "missing":
            del altered["main.tex"]
        elif change == "extra":
            altered["unexpected.txt"] = b"extra"
        if change == "truncate":
            original(path, kind, files)
            payload = path.read_bytes()
            path.write_bytes(payload[:len(payload) // 2])
        elif change in ("bytes", "missing", "extra"):
            original(path, kind, altered)
        elif kind == "zip":
            original(path, kind, files)
            with zipfile.ZipFile(path, "a") as archive:
                info = zipfile.ZipInfo("main.tex" if change == "duplicate" else "link")
                info.external_attr = (stat.S_IFREG if change == "duplicate" else stat.S_IFLNK) << 16
                archive.writestr(info, files["main.tex"])
        else:
            with tarfile.open(path, "x:gz") as archive:
                for name, payload in [*files.items(), ("main.tex" if change == "duplicate" else "link", b"bad")]:
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    if name == "link":
                        info.type, info.linkname = tarfile.SYMTYPE, "main.tex"
                    archive.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr(packager, "write_archive", corrupt)
    output = tmp_path / ("corrupt" + suffix)
    if change == "duplicate" and suffix == ".zip":
        with pytest.warns(UserWarning, match="Duplicate name"):
            assert run(paper, output) == 1
    else:
        assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("winner", ["archive", "sidecar"])
def test_concurrent_publisher_not_overwritten(paper, tmp_path, monkeypatch, winner):
    original = os.link
    output = tmp_path / "race.zip"
    sidecar = Path(str(output) + ".manifest.json")
    target = output if winner == "archive" else sidecar

    def competing_link(source, destination):
        if destination == target:
            target.write_bytes(b"concurrent writer")
        return original(source, destination)
    monkeypatch.setattr(packager.os, "link", competing_link)
    assert run(paper, output) == 1
    assert target.read_bytes() == b"concurrent writer"
    other = sidecar if winner == "archive" else output
    assert not other.exists()
    assert not list(tmp_path.glob(".prigm-package-*"))


def test_published_corruption_detected_before_completion(paper, tmp_path, monkeypatch):
    original = os.link
    output = tmp_path / "post-link.zip"

    def corrupt_link(source, destination):
        original(source, destination)
        if destination == output:
            destination.write_bytes(b"corrupt")
    monkeypatch.setattr(packager.os, "link", corrupt_link)
    assert run(paper, output) == 1
    assert_nothing(output)


def test_rollback_preserves_replacement_by_another_writer(paper, tmp_path, monkeypatch):
    original = packager.verify_archive
    output = tmp_path / "replaced.zip"

    def replace_on_readback(path, kind, files):
        if path == output:
            output.unlink()
            output.write_bytes(b"unfamiliar replacement")
            raise ValueError("concurrent replacement")
        return original(path, kind, files)
    monkeypatch.setattr(packager, "verify_archive", replace_on_readback)
    assert run(paper, output) == 1
    assert output.read_bytes() == b"unfamiliar replacement"
    assert not Path(str(output) + ".manifest.json").exists()
    assert not list(tmp_path.glob(".prigm-package-*"))


@pytest.mark.parametrize("leak", ["filename", "dtype-title", "pdfauthor"])
def test_identity_metadata_is_not_exempt(paper, tmp_path, leak):
    if leak == "filename":
        save_json(paper / "evidence/toy/Alison.json", {"value": 1})
    elif leak == "dtype-title":
        array = np.zeros(1, dtype=[(("Kevin Lee", "value"), "f8")])
        np.savez(paper / "evidence/toy/toy.npz", values=array)
    else:
        source = paper / "main.tex"
        source.write_text(source.read_text() + r"\hypersetup{pdfauthor={Some Author}}")
    output = tmp_path / "metadata.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("addition", ["article.cls", "environ.sty", "geometry.sty", "keyval.sty", "graphics.cfg",
                                      "sections/part", "evidence/toy/extra.json",
                                      "evidence/observations/extra.json", "evidence/sensitivity/extra.json"])
def test_dependency_addition_during_verifier_refused(paper, tmp_path, monkeypatch, addition):
    if addition == "sections/part":
        (paper / "sections").mkdir()
        (paper / "sections/part.tex").write_text("A section.")
        source = paper / "main.tex"
        source.write_text(source.read_text() + r"\input{sections/part}")
    original = packager.subprocess.run

    def add_dependency(*args, **kwargs):
        result = original(*args, **kwargs)
        (paper / addition).write_text("{}")
        return result
    monkeypatch.setattr(packager.subprocess, "run", add_dependency)
    output = tmp_path / "changed-closure.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("name", ["environ.sty", "geometry.sty", "keyval.sty", "graphics.cfg", "unsrtnat.bst"])
def test_vendor_style_dependency_shadow_refused(paper, tmp_path, name):
    (paper / name).write_text("\\AtBeginDocument{LOCAL STYLE MARKER\\par}\n")
    output = tmp_path / "shadow.zip"
    with pytest.raises(ValueError, match="local system-TeX shadow"):
        packager.package(paper, output)
    assert_nothing(output)


def test_verifier_replacement_detected(paper, tmp_path, monkeypatch):
    original = packager.subprocess.run

    def replace_verifier(*args, **kwargs):
        result = original(*args, **kwargs)
        replacement = tmp_path / "replacement-verifier.py"
        replacement.write_bytes(packager.VERIFIER.read_bytes())
        replacement.replace(packager.VERIFIER)
        return result
    monkeypatch.setattr(packager.subprocess, "run", replace_verifier)
    output = tmp_path / "changed-gate.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


def test_cli_help_is_local_and_has_no_bypass():
    result = subprocess.run([sys.executable, "-B", str(REPO / "scripts/package_prigm_paper.py"), "--help"],
                            text=True, capture_output=True, check=True)
    assert "--paper-dir" in result.stdout and "--output" in result.stdout
    assert "--skip" not in result.stdout and "--allow" not in result.stdout


@pytest.mark.parametrize("name", [
    *["evidence/data/" + n for n in (*packager.DATA_RECORDS, "MANIFEST.json")],
    *["evidence/real/" + n for n in packager.REAL_FILES],
    *["evidence/toy/learned/" + n for n in packager.LEARNED_FILES],
    *["evidence/toy/" + n for n in packager.TOY_FILES],
    "evidence/observations/test.npz", "evidence/observations/MANIFEST.json",
    "evidence/sensitivity/training-strata.json", "evidence/sensitivity/MANIFEST.json",
])
def test_all_declared_public_evidence_is_required(paper, tmp_path, name):
    (paper / name).unlink()
    output = tmp_path / "missing-evidence.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("group", ["", "data", "real", "toy", "toy/learned", "observations", "sensitivity"])
@pytest.mark.parametrize("extra", ["file", "directory"])
def test_evidence_whitelist_rejects_extra_records(paper, tmp_path, group, extra):
    path = paper / "evidence" / group / "extra.json"
    if extra == "file":
        save_json(path, {})
    else:
        path.mkdir()
    output = tmp_path / "extra-evidence.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("damage", ["files", "records", "original_name", "invalid_original_hash",
                                    "inventory_not_identical", "reduced_hash_conflated", "public_bytes"])
def test_packaged_data_original_and_public_bindings(paper, tmp_path, damage):
    path = paper / "evidence/data/MANIFEST.json"
    manifest = json.loads(path.read_text())
    if damage in ("files", "records"):
        del manifest[damage]["quality.json"]
    elif damage == "original_name":
        manifest["records"]["quality.json"]["source_record"] = "quality.json"
    elif damage == "invalid_original_hash":
        manifest["records"]["q15.json"]["source_sha256"] = "bad"
    elif damage == "inventory_not_identical":
        manifest["records"]["inventory.json"]["source_sha256"] = "0" * 64
    elif damage == "reduced_hash_conflated":
        manifest["records"]["quality.json"]["source_sha256"] = manifest["files"]["quality.json"]
    else:
        (paper / "evidence/data/q15.json").write_bytes(b"{}")
    save_json(path, manifest)
    output = tmp_path / "data-binding.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("name", ["evidence/real/gradients.json", "evidence/data/quality.json",
                                  "evidence/toy/learned/learned.json", "evidence/toy/learned/reference-calibration.json",
                                  "evidence/toy/learned/MANIFEST.json", "evidence/observations/MANIFEST.json",
                                  "evidence/sensitivity/training-strata.json", "evidence/sensitivity/MANIFEST.json"])
def test_new_payloads_are_privacy_and_finiteness_scanned(paper, tmp_path, name):
    path = paper / name
    for payload in (r'{"path":"\u002fhome\u002fprivate"}', '{"value":NaN}', '{"value":1,"value":2}'):
        path.write_text(payload)
        output = tmp_path / "unsafe-payload.zip"
        assert run(paper, output) == 1
        assert_nothing(output)


@pytest.mark.parametrize("name", ["evidence/data", "evidence/toy/learned", "evidence/toy/learned/learned.json",
                                  "evidence/observations", "evidence/observations/test.npz",
                                  "evidence/observations/MANIFEST.json", "evidence/sensitivity",
                                  "evidence/sensitivity/training-strata.json", "evidence/sensitivity/MANIFEST.json"])
def test_new_evidence_symlinks_are_not_followed(paper, tmp_path, name):
    path = paper / name
    outside = tmp_path / "outside-evidence"
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=outside.is_dir())
    output = tmp_path / "evidence-link.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


@pytest.mark.parametrize("name", ["evidence/toy/learned/learned.json",
                                  "evidence/toy/learned/reference-calibration.json",
                                  "evidence/observations/test.npz",
                                  "evidence/sensitivity/training-strata.json"])
def test_new_payload_hashes_are_bound_before_verifier(paper, tmp_path, name):
    path = paper / name
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "new-payload-hash.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


def rebind_fixture_payload(path):
    """Rehash deliberately tampered fixtures to exercise cross-record bindings."""
    manifest_path = path.parent / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    field = "output_sha256" if "output_sha256" in manifest else "files"
    manifest[field][path.name] = sha(path.read_bytes())
    if field == "files":
        manifest["records"][path.name]["source_sha256"] = manifest[field][path.name]
    save_json(manifest_path, manifest)


@pytest.mark.parametrize("group,name", [("observations", "test.npz"),
                                        ("sensitivity", "training-strata.json")])
@pytest.mark.parametrize("damage", ["schema", "missing_files", "extra_files", "missing_records",
                                    "extra_records", "record_field", "source_name", "source_hash",
                                    "public_hash", "protocol", "missing_protocol"])
def test_unchanged_evidence_manifest_bindings(paper, tmp_path, group, name, damage):
    path = paper / "evidence" / group / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    if damage == "schema":
        manifest["schema_version"] = True
    elif damage.startswith("missing_") and damage != "missing_protocol":
        del manifest[damage.removeprefix("missing_")][name]
    elif damage.startswith("extra_"):
        manifest[damage.removeprefix("extra_")]["extra.json"] = "0" * 64
    elif damage == "record_field":
        manifest["records"][name]["extra"] = "undocumented"
    elif damage == "source_name":
        manifest["records"][name]["source_record"] = name
    elif damage == "source_hash":
        manifest["records"][name]["source_sha256"] = "0" * 64
    elif damage == "public_hash":
        manifest["files"][name] = "0" * 64
    elif damage == "protocol":
        manifest["source_protocol_sha256"] = "0" * 64
    else:
        del manifest["source_protocol_sha256"]
    save_json(path, manifest)
    output = tmp_path / "unchanged-binding.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


def test_observation_rehash_cannot_bypass_protocol_binding(paper, tmp_path, capsys):
    path = paper / "evidence/observations/test.npz"
    path.write_bytes(path.read_bytes() + b" ")
    rebind_fixture_payload(path)
    output = tmp_path / "observation-protocol.zip"
    assert run(paper, output) == 1
    assert "observation/protocol TEST SHA256 mismatch" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("field", ["schema_version", "protocol_sha256", "input_protocol", "TRAIN", "TEST"])
def test_sensitivity_rehash_cannot_bypass_real_identity(paper, tmp_path, capsys, field):
    path = paper / "evidence/sensitivity/training-strata.json"
    record = json.loads(path.read_text())
    if field in ("TRAIN", "TEST"):
        record["arrays"][field]["sha256"] = "0" * 64
    elif field == "input_protocol":
        record["input_sha256"]["PROTOCOL.json"] = "0" * 64
    else:
        record[field] = True if field == "schema_version" else "0" * 64
    save_json(path, record)
    rebind_fixture_payload(path)
    output = tmp_path / "sensitivity-protocol.zip"
    assert run(paper, output) == 1
    assert "sensitivity real protocol/array identity mismatch" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("damage", ["missing", "extra", "object", "identity"])
def test_observation_npz_is_scanned_before_hash_checks(paper, tmp_path, capsys, damage):
    path = paper / "evidence/observations/test.npz"
    with np.load(path, allow_pickle=False) as blob:
        arrays = {key: blob[key] for key in blob.files}
    if damage == "missing":
        del arrays["mean"]
    elif damage == "extra":
        arrays["extra"] = np.zeros(1)
    elif damage == "object":
        arrays["ids"] = np.array([{"private": "value"}], dtype=object)
    else:
        arrays["ids"] = np.array(["/home/private/run"])
    np.savez(path, **arrays)
    output = tmp_path / "observation-fields.zip"
    assert run(paper, output) == 1
    error = capsys.readouterr().err
    assert ("field inventory mismatch" if damage in ("missing", "extra") else
            "Object arrays cannot be loaded" if damage == "object" else "author/path leak") in error
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name,keys,value", [
    ("MANIFEST.json", ("schema_version",), True),
    ("learned.json", ("schema_version",), True),
    ("reference-calibration.json", ("schema_version",), True),
    ("MANIFEST.json", ("output_sha256", "extra.json"), "0" * 64),
    ("MANIFEST.json", ("input_sha256", "PROTOCOL.json"), "0" * 64),
    ("MANIFEST.json", ("input_sha256", "artifacts/toy.npz"), "0" * 64),
    ("MANIFEST.json", ("input_sha256", "TRAINED.json"), "0" * 64),
    ("MANIFEST.json", ("input_sha256", "EVALUATED.json"), "0" * 64),
    ("MANIFEST.json", ("input_sha256", "extra.json"), "bad"),
    ("MANIFEST.json", ("input_sha256", "../outside.json"), "0" * 64),
    ("MANIFEST.json", ("exporter_source_sha256",), "bad"),
    ("MANIFEST.json", ("sources", "coordinator_sha256"), "0" * 64),
    ("learned.json", ("original_sha256", "protocol"), "0" * 64),
    ("learned.json", ("sources", "training", "prigm/train.py"), "0" * 64),
    ("learned.json", ("arrays", "sha256"), "0" * 64),
    ("reference-calibration.json", ("source_protocol_sha256",), "0" * 64),
    ("reference-calibration.json", ("source_oracle_sha256",), "0" * 64),
    ("reference-calibration.json", ("arrays", "sha256"), "0" * 64),
    ("reference-calibration.json", ("oracle_metadata", "toy_eval_sha256"), "0" * 64),
])
def test_learned_manifest_and_public_bindings(paper, tmp_path, name, keys, value):
    path = paper / "evidence/toy/learned" / name
    record = json.loads(path.read_text())
    target = record
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    save_json(path, record)
    if name != "MANIFEST.json":
        rebind_fixture_payload(path)
    output = tmp_path / "learned-binding.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("field", ["output_sha256", "input_sha256", "original_sha256", "sources"])
def test_learned_manifest_requires_binding_inventories(paper, tmp_path, field):
    path = paper / "evidence/toy/learned/MANIFEST.json"
    manifest = json.loads(path.read_text())
    del manifest[field]
    save_json(path, manifest)
    output = tmp_path / "learned-missing-inventory.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("key", ["protocol", "oracle", "array", "trained", "evaluated"])
def test_consistent_learned_originals_still_bind_external_records(paper, tmp_path, capsys, key):
    directory = paper / "evidence/toy/learned"
    for name in ("MANIFEST.json", "learned.json"):
        path = directory / name
        record = json.loads(path.read_text())
        record["original_sha256"][key] = "0" * 64
        save_json(path, record)
    rebind_fixture_payload(directory / "learned.json")
    output = tmp_path / "learned-original.zip"
    assert run(paper, output) == 1
    assert ("public oracle/array/protocol binding mismatch" if key == "oracle" else
            "original/input SHA256 mismatch") in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name", ["oracle-stationary-01.json", "hmm-stationary-01.json", "toy.npz"])
def test_learned_bindings_check_public_stationary_files(paper, tmp_path, capsys, name):
    path = paper / "evidence/toy" / name
    if name == "hmm-stationary-01.json":
        record = json.loads(path.read_text())
        record["toy_eval_sha256"] = "0" * 64
        save_json(path, record)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "stationary-binding.zip"
    assert run(paper, output) == 1
    assert ("stationary toy array SHA256 mismatch" if name.startswith("hmm") else
            "learned toy public oracle/array/protocol binding mismatch") in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


def test_fifth_real_payload_hash_is_bound_before_verifier(paper, tmp_path):
    path = paper / "evidence/real/gradients.json"
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "gradient-hash.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_deadline_extension_roundtrip_preserves_old_groups(paper, tmp_path, suffix, separate, fixed):
    before, _, _ = packager.collect(paper)
    assert len(before) == 28
    payloads = deadline_fixture(paper, separate=separate, fixed=fixed)
    figure(paper, "operator", "evidence/deadline/operator-mechanism.json")
    source = paper / "main.tex"
    source.write_text(source.read_text() + "\nFull-flank Gaussian\n"
                      + r"\includegraphics{figures/operator/fig_operator.pdf}"
                      + (r"\label{tab:fixedfinal}" if fixed else ""))
    files, _, evidence = packager.collect(paper)
    assert len(files) == 45 + 2 * separate + fixed
    assert set(evidence) >= {"evidence/deadline/" + n for n in payloads}
    assert all(files[n] == p for n, p in before.items() if n != "main.tex")
    assert {n.removeprefix("evidence/deadline/"): p for n, p in files.items()
            if n.startswith("evidence/deadline/")} == payloads
    output = tmp_path / ("extended" + suffix)
    assert run(paper, output) == 0
    assert read_archive(output) == files
    assert all("WRITE.lock" not in n and "LATEST" not in n for n in files)
    assert not any(name.endswith((".pt", ".pth")) for name in files)
    assert "main.pdf" not in files


@pytest.mark.parametrize("lam", [0.001, 0.1, 1])
@pytest.mark.parametrize("rho", [0, 0.5, 1])
def test_gaussian_candidate_names_are_generated_from_the_fixed_grid(lam, rho):
    name = f"lambda_{lam:g}_rho_{rho:g}"
    row = {"lambda": lam, "rho": rho, "month_balanced_primary_es": 1.5}
    assert packager.gaussian_selected({"selected": {"raw": row, "corrected": row},
                                       "selected_union": [name]}) == [name]


def test_gaussian_negative_zero_cannot_introduce_an_alternate_directory():
    row = {"lambda": 0.1, "rho": -0.0, "month_balanced_primary_es": 1.5}
    with pytest.raises(ValueError, match="selected_union"):
        packager.gaussian_selected({"selected": {"raw": row, "corrected": row},
                                   "selected_union": ["lambda_0.1_rho_-0"]})


@pytest.mark.parametrize("damage", ["input", "generator", "pdf", "ambiguous_input", "missing_png"])
def test_operator_figure_uses_existing_companion_and_input_bindings(paper, tmp_path, damage):
    deadline_fixture(paper)
    figure(paper, "operator", "evidence/deadline/operator-mechanism.json")
    source = paper / "main.tex"
    source.write_text(source.read_text() + r"\includegraphics{figures/operator/fig_operator.pdf}")
    directory = paper / "figures/operator"
    if damage == "missing_png":
        (directory / "fig_operator.png").rename(tmp_path / "unpackaged-operator.png")
    elif damage in ("generator", "pdf"):
        path = paper / "fig_operator.py" if damage == "generator" else directory / "fig_operator.pdf"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = directory / "fig_operator.manifest.json"
        manifest = json.loads(path.read_text())
        if damage == "input":
            manifest["inputs"]["record"]["sha256"] = "0" * 64
        else:
            manifest["inputs"]["record"]["path"] = "MANIFEST.json"
        save_json(path, manifest)
    output = tmp_path / "bad-operator-figure.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


def test_deadline_optional_without_new_claims_and_reduced_original_hash(paper, tmp_path):
    deadline_fixture(paper, fixed=True)
    manifest_path = paper / "evidence/deadline/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["original_sha256"]["fixed-final.json"] = sha(b"unredacted summary")
    save_json(manifest_path, manifest)
    output = tmp_path / "optional.zip"
    assert run(paper, output) == 0
    assert "evidence/deadline/fixed-final.json" in read_archive(output)


@pytest.mark.parametrize("claim", [r"\includegraphics{figures/operator/fig_operator.pdf}",
                                  "Full-flank Gaussian & 1.0", "Full-flank\nGaussian & 1.0",
                                  "Gaussian regression & 1.0", "Gaussian\nregression & 1.0",
                                  r"\label{tab:fixedfinal}"])
def test_new_claims_require_deadline_data_before_verifier(paper, tmp_path, capsys, claim):
    source = paper / "main.tex"
    source.write_text(source.read_text() + "\n" + claim)
    output = tmp_path / "unsupported-claims.zip"
    assert run(paper, output) == 1
    assert "new deadline claims require evidence/deadline" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


def test_fixed_final_table_requires_its_optional_payload(paper, tmp_path, capsys):
    deadline_fixture(paper)
    source = paper / "main.tex"
    source.write_text(source.read_text() + r"\label{tab:fixedfinal}")
    output = tmp_path / "missing-fixed-final.zip"
    assert run(paper, output) == 1
    assert "fixed-final claims require" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


DEADLINE_FILES = ["MANIFEST.json", "operator-mechanism.json", "fixed-final.json",
    *["gaussian/" + n for n in ("EXTENSION.json", "FIT.npz", "FIT.json", "VALIDATION.json",
                               "SELECTION.json", "FIT_SELECT.json")],
    *["gaussian/test/" + n for n in ("TEST_STARTED.json", "COMPLETED.json", "SCORES.json")],
    *["gaussian/test/lambda_0.1_rho_0/gen_00000000/" + n for n in ("COMMIT.json", "ensemble.npz")]]


@pytest.mark.parametrize("name", DEADLINE_FILES)
def test_deadline_missing_payload_refused(paper, tmp_path, name):
    deadline_fixture(paper, fixed=True, omit={name})
    output = tmp_path / "missing-extension.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name", ["extra.json", "fixed-final.json", "model.pt", "execution.log",
    "gaussian/extra.json", "gaussian/WRITE.lock", "gaussian/LATEST", "gaussian/test/extra.json",
    "gaussian/test/lambda_1_rho_1", "gaussian/test/lambda_0.1_rho_0/WRITE.lock",
    "gaussian/test/lambda_0.1_rho_0/LATEST", "gaussian/test/lambda_0.1_rho_0/gen_00000001",
    "gaussian/test/lambda_0.1_rho_0/gen_00000000/extra.json"])
@pytest.mark.parametrize("directory", [False, True])
def test_deadline_extra_files_and_directories_are_refused_not_removed(paper, tmp_path, name, directory):
    deadline_fixture(paper)
    path = paper / "evidence/deadline" / name
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b"unfamiliar bytes")
    output = tmp_path / "extra-extension.zip"
    assert run(paper, output) == 1
    assert path.is_dir() if directory else path.read_bytes() == b"unfamiliar bytes"
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name", ["", "gaussian", "gaussian/test", "gaussian/test/lambda_0.1_rho_0",
    "gaussian/test/lambda_0.1_rho_0/gen_00000000", *DEADLINE_FILES])
def test_deadline_symlinks_refused(paper, tmp_path, name):
    deadline_fixture(paper, fixed=True)
    path = paper / "evidence/deadline" / name
    outside = tmp_path / "outside-deadline"
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=outside.is_dir())
    output = tmp_path / "symlink-extension.zip"
    assert run(paper, output) == 1
    assert path.is_symlink() and outside.exists()
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("damage", ["missing_kind", "extra_kind", "lambda", "rho", "bool", "string",
    "extra_field", "missing_field", "empty_union", "duplicate_union", "extra_union", "wrong_union",
    "traversal", "noncanonical", "reverse_union", "union_not_list"])
def test_deadline_selected_union_is_only_the_fixed_grid(paper, tmp_path, capsys, damage):
    deadline_fixture(paper, separate=damage == "reverse_union")
    path = paper / "evidence/deadline/gaussian/SELECTION.json"
    selection = json.loads(path.read_text())
    row = selection["selected"]["raw"]
    if damage == "missing_kind":
        del selection["selected"]["corrected"]
    elif damage == "extra_kind":
        selection["selected"]["another"] = row
    elif damage in ("lambda", "rho", "bool", "string"):
        row["rho" if damage == "rho" else "lambda"] = {
            "lambda": 0.01, "rho": 0.25, "bool": True, "string": "0.1"}[damage]
    elif damage == "extra_field":
        row["path"] = "unlisted"
    elif damage == "missing_field":
        del row["rho"]
    else:
        selection["selected_union"] = {
            "empty_union": [], "duplicate_union": ["lambda_0.1_rho_0"] * 2,
            "extra_union": ["lambda_0.001_rho_0", "lambda_0.1_rho_0", "lambda_1_rho_1"],
            "wrong_union": ["lambda_1_rho_1"], "traversal": ["../outside"],
            "noncanonical": ["lambda_0.10_rho_0.0"],
            "reverse_union": ["lambda_1_rho_1", "lambda_0.1_rho_0"],
            "union_not_list": "lambda_0.1_rho_0"}[damage]
    save_json(path, selection)
    rehash_deadline(paper)
    output = tmp_path / "bad-selection.zip"
    assert run(paper, output) == 1
    assert "Gaussian select" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("damage", ["schema", "protocol", "missing_payload", "extra_payload", "manifest_payload",
    "unsafe_payload", "public_hash", "operator_original", "fit_select_original", "bad_fixed_original",
    "missing_original", "extra_original", "producer_hash", "producer_path", "empty_producers",
    "transformation", "missing_transformation", "extra_transformation", "undeclared_fixed"])
def test_deadline_manifest_is_not_an_open_ended_whitelist(paper, tmp_path, damage):
    deadline_fixture(paper, fixed=True)
    path = paper / "evidence/deadline/MANIFEST.json"
    manifest = json.loads(path.read_text())
    if damage == "schema":
        manifest["schema_version"] = True
    elif damage == "protocol":
        manifest["source_protocol_sha256"] = "0" * 64
    elif damage == "missing_payload":
        del manifest["files"]["operator-mechanism.json"]
    elif damage in ("extra_payload", "manifest_payload", "unsafe_payload"):
        manifest["files"][{"extra_payload": "gaussian/unknown.json", "manifest_payload": "MANIFEST.json",
                           "unsafe_payload": "../outside.json"}[damage]] = "0" * 64
    elif damage == "public_hash":
        manifest["files"]["operator-mechanism.json"] = "0" * 64
    elif damage in ("operator_original", "fit_select_original", "bad_fixed_original"):
        manifest["original_sha256"][{"operator_original": "operator-mechanism.json",
            "fit_select_original": "gaussian/FIT_SELECT.json", "bad_fixed_original": "fixed-final.json"}[damage]] = (
                "bad" if damage == "bad_fixed_original" else "0" * 64)
    elif damage == "missing_original":
        del manifest["original_sha256"]["fixed-final.json"]
    elif damage == "extra_original":
        manifest["original_sha256"]["unknown.json"] = "0" * 64
    elif damage == "producer_hash":
        manifest["producers"]["analysis/prigm_full_flank_gaussian.py"] = "bad"
    elif damage == "producer_path":
        manifest["producers"]["../outside.py"] = "0" * 64
    elif damage == "empty_producers":
        manifest["producers"] = {}
    elif damage == "transformation":
        manifest["transformations"]["gaussian"] = "changed values"
    elif damage == "missing_transformation":
        del manifest["transformations"]["gaussian"]
    elif damage == "extra_transformation":
        manifest["transformations"]["unknown.json"] = "byte-identical"
    else:
        del manifest["files"]["fixed-final.json"]
    save_json(path, manifest)
    output = tmp_path / "bad-deadline-manifest.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name", [n for n in DEADLINE_FILES if n != "MANIFEST.json"])
def test_deadline_payload_bytes_bound_before_verifier(paper, tmp_path, name):
    deadline_fixture(paper, fixed=True)
    path = paper / "evidence/deadline" / name
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "deadline-bytes.zip"
    assert run(paper, output) == 1
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name,field,value", [
    ("FIT_SELECT.json", "files", {"FIT.npz": "0" * 64}),
    ("FIT_SELECT.json", "source_protocol_sha256", "0" * 64),
    ("FIT_SELECT.json", "phase", "OTHER"),
    ("FIT_SELECT.json", "schema_version", True),
    ("FIT.json", "fit_payload_sha256", "0" * 64),
    ("FIT.json", "fit_payload", "other.npz"),
    ("FIT.json", "extension_protocol_sha256", "0" * 64),
    ("VALIDATION.json", "fit_payload_sha256", "0" * 64),
    ("SELECTION.json", "fit_payload_sha256", "0" * 64),
    ("SELECTION.json", "validation_sha256", "0" * 64),
    ("EXTENSION.json", "source_protocol_sha256", "0" * 64),
    ("test/TEST_STARTED.json", "fit_select_sha256", "0" * 64),
    ("test/TEST_STARTED.json", "fit_payload_sha256", "0" * 64),
    ("test/TEST_STARTED.json", "selected_union", []),
    ("test/SCORES.json", "selection_sha256", "0" * 64),
    ("test/SCORES.json", "candidates", {}),
    ("test/COMPLETED.json", "files", {"test/SCORES.json": "0" * 64}),
    ("test/COMPLETED.json", "selected_union", []),
    ("test/COMPLETED.json", "fit_select_sha256", "0" * 64),
])
def test_deadline_rehashed_manifest_cannot_bypass_inner_bindings(paper, tmp_path, capsys, name, field, value):
    deadline_fixture(paper)
    directory = paper / "evidence/deadline/gaussian"
    path = directory / name
    record = json.loads(path.read_text())
    record[field] = value
    save_json(path, record)
    if name in packager.GAUSSIAN_FIT_FILES:
        receipt_path = directory / "FIT_SELECT.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][name] = sha(path.read_bytes())
        save_json(receipt_path, receipt)
    rehash_deadline(paper)
    output = tmp_path / "inner-binding.zip"
    assert run(paper, output) == 1
    assert "Gaussian" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name,error", [("FIT.npz", "fit payload SHA256"),
    ("test/lambda_0.1_rho_0/gen_00000000/ensemble.npz", "COMMIT/ensemble SHA256")])
def test_rehashed_gaussian_npz_still_needs_internal_hash_binding(paper, tmp_path, capsys, name, error):
    deadline_fixture(paper)
    directory = paper / "evidence/deadline/gaussian"
    path = directory / name
    path.write_bytes(path.read_bytes() + b" ")
    if name == "FIT.npz":
        receipt_path = directory / "FIT_SELECT.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"][name] = sha(path.read_bytes())
        save_json(receipt_path, receipt)
    rehash_deadline(paper)
    output = tmp_path / "inner-npz.zip"
    assert run(paper, output) == 1
    assert error in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("field,value", [("files", {"other.npz": "0" * 64}), ("generation", True),
    ("control_fit_sha256", "0" * 64), ("selection_sha256", "0" * 64), ("fit_select_sha256", "0" * 64)])
def test_rehashed_selected_commit_cannot_bypass_ensemble_bindings(paper, tmp_path, capsys, field, value):
    deadline_fixture(paper)
    directory = paper / "evidence/deadline/gaussian"
    name = "test/lambda_0.1_rho_0/gen_00000000/COMMIT.json"
    path = directory / name
    commit = json.loads(path.read_text())
    commit[field] = value
    save_json(path, commit)
    scores = json.loads((directory / "test/SCORES.json").read_text())
    scores["candidates"]["lambda_0.1_rho_0"]["ensemble_commit_sha256"] = sha(path.read_bytes())
    save_json(directory / "test/SCORES.json", scores)
    completed = json.loads((directory / "test/COMPLETED.json").read_text())
    for member in completed["files"]:
        completed["files"][member] = sha((directory / member).read_bytes())
    save_json(directory / "test/COMPLETED.json", completed)
    rehash_deadline(paper)
    output = tmp_path / "inner-commit.zip"
    assert run(paper, output) == 1
    assert "COMMIT/ensemble SHA256" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name", DEADLINE_FILES)
def test_deadline_private_strings_are_scanned_even_when_rehashed(paper, tmp_path, capsys, name):
    deadline_fixture(paper, fixed=True)
    path = paper / "evidence/deadline" / name
    if name.endswith(".npz"):
        with np.load(path, allow_pickle=False) as blob:
            arrays = {key: blob[key] for key in blob.files}
        arrays["private_path"] = np.array(["/home/private/study"])
        np.savez(path, **arrays)
    else:
        record = json.loads(path.read_text())
        record["private_path"] = "/home/private/study"
        save_json(path, record)
    rehash_deadline(paper)
    output = tmp_path / "private-extension.zip"
    assert run(paper, output) == 1
    assert "author/path leak" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("name", ["gaussian/FIT.npz", "gaussian/test/lambda_0.1_rho_0/gen_00000000/ensemble.npz"])
def test_gaussian_object_npz_is_never_unpickled(paper, tmp_path, capsys, name):
    deadline_fixture(paper)
    np.savez(paper / "evidence/deadline" / name, values=np.array([{"hidden": 1}], dtype=object))
    rehash_deadline(paper)
    output = tmp_path / "object-gaussian.zip"
    assert run(paper, output) == 1
    assert "Object arrays cannot be loaded" in capsys.readouterr().err
    assert_nothing(output)
    assert not packager.VERIFIER.with_suffix(".called.json").exists()


@pytest.mark.parametrize("change", ["payload", "extra", "lock"])
def test_deadline_changes_during_verifier_preserve_atomic_refusal(paper, tmp_path, monkeypatch, change):
    deadline_fixture(paper)
    original = packager.subprocess.run

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        name = {"payload": "operator-mechanism.json", "extra": "gaussian/extra.json",
                "lock": "gaussian/test/lambda_0.1_rho_0/WRITE.lock"}[change]
        (paper / "evidence/deadline" / name).write_bytes(b"{}")
        return result
    monkeypatch.setattr(packager.subprocess, "run", mutate)
    output = tmp_path / "changed-deadline.zip"
    assert run(paper, output) == 1
    assert_nothing(output)


def illustration_fixture(paper):
    """Packaging-only synthetic bytes; the final scientific gate is separate."""
    deadline_fixture(paper, fixed=True)
    directory = paper / "evidence/imputation"
    directory.mkdir()
    np.savez(directory / "sample-excerpt.npz", truth=np.zeros((2, 48, 6)),
             samples=np.zeros((4, 2, 39, 12, 5)), ids=np.array(["first", "second"]),
             methods=np.array(["D", "O-S", "G5", "full_flank_gaussian"]))
    excerpt_hash = sha((directory / "sample-excerpt.npz").read_bytes())
    save_json(directory / "MANIFEST.json", {"schema_version": 1,
              "files": {"sample-excerpt.npz": excerpt_hash}})
    figure(paper, "fixed_final", "evidence/deadline/fixed-final.json")
    source = paper / "fig_imputation.py"
    source.write_text('"""Anonymous synthetic illustration generator."""\n')
    for name in ("case_120", "case_360", "detail"):
        target = paper / "figures/imputation" / name
        target.mkdir(parents=True)
        pdf, png = target / "fig_imputation.pdf", target / "fig_imputation.png"
        pdf.write_bytes(b"%PDF-1.4\nsynthetic\n%%EOF\n")
        png.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
        save_json(target / "fig_imputation.manifest.json", {
            "schema_version": 1, "figure": "fig_imputation",
            "source": {"path": source.name, "sha256": sha(source.read_bytes())},
            "inputs": {"excerpt": {"path": "evidence/imputation/sample-excerpt.npz", "sha256": excerpt_hash},
                       "provenance": {"path": "evidence/imputation/MANIFEST.json",
                                      "sha256": sha((directory / "MANIFEST.json").read_bytes())}},
            "output_sha256": {p.name: sha(p.read_bytes()) for p in (pdf, png)}})
    tex = (paper / "main.tex").read_text()
    graphics = "\n".join(r"\includegraphics{" + name + "}" for name in sorted(packager.ILLUSTRATION_FIGURES))
    (paper / "main.tex").write_text(tex.replace(r"\end{document}", graphics + "\n" + r"\end{document}"))


def test_illustration_family_collects_only_declared_companions(paper):
    illustration_fixture(paper)
    files, identities, evidence = packager.collect(paper)
    assert packager.ILLUSTRATION_FIGURES.issubset(files)
    assert {"evidence/imputation/MANIFEST.json", "evidence/imputation/sample-excerpt.npz"}.issubset(evidence)
    assert "fig_imputation.py" in files and "fig_fixed_final.py" in files
    assert set(files) == set(identities)


@pytest.mark.parametrize("name", sorted(packager.ILLUSTRATION_FIGURES))
def test_illustration_family_cannot_drop_one_image(paper, name):
    illustration_fixture(paper)
    tex = (paper / "main.tex").read_text().replace(r"\includegraphics{" + name + "}", "")
    (paper / "main.tex").write_text(tex)
    with pytest.raises(ValueError, match="family is incomplete"):
        packager.collect(paper)


@pytest.mark.parametrize("field", ["extra", "missing"])
def test_imputation_npz_has_exact_fields(field):
    fields = {"truth": np.zeros((2, 48, 6)), "samples": np.zeros((4, 2, 39, 12, 5)),
              "ids": np.array(["first", "second"]), "methods": np.array(["D", "O-S", "G5", "full_flank_gaussian"])}
    if field == "extra":
        fields["unrequested"] = np.ones(1)
    else:
        fields.pop("methods")
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    with pytest.raises(ValueError, match="imputation NPZ field inventory"):
        packager.check_npz(buffer.getvalue(), "evidence/imputation/sample-excerpt.npz")


@pytest.mark.parametrize("damage", ["excerpt_hash", "extra_file", "ambiguous_manifest", "undeclared_case"])
def test_illustration_evidence_and_inputs_are_closed(paper, damage):
    illustration_fixture(paper)
    directory = paper / "evidence/imputation"
    if damage == "excerpt_hash":
        save_json(directory / "MANIFEST.json", {"files": {"sample-excerpt.npz": "0" * 64}})
    elif damage == "extra_file":
        (directory / "extra.json").write_text("{}")
    elif damage == "ambiguous_manifest":
        manifest_path = paper / "figures/imputation/case_120/fig_imputation.manifest.json"
        value = json.loads(manifest_path.read_text())
        value["inputs"]["provenance"]["path"] = "MANIFEST.json"
        save_json(manifest_path, value)
    else:
        tex = (paper / "main.tex").read_text().replace("case_120/fig_imputation.pdf", "case_121/fig_imputation.pdf")
        (paper / "main.tex").write_text(tex)
    with pytest.raises(ValueError):
        packager.collect(paper)


COMPANION_URL = "https://github.com/magnaprog/diffusion-consistency"


@pytest.mark.parametrize("name", ["main.tex", "appendix.tex", "references.bib", "evidence/record.json", "fig_hmm.py"])
@pytest.mark.parametrize("camera_ready", [False, True])
def test_companion_url_scope(name, camera_ready):
    if camera_ready and name == "main.tex":
        packager.scan_text(COMPANION_URL, name, camera_ready=camera_ready)
    else:
        with pytest.raises(ValueError, match="author/path leak"):
            packager.scan_text(COMPANION_URL, name, camera_ready=camera_ready)


@pytest.mark.parametrize("text", [
    COMPANION_URL, r"\url{" + COMPANION_URL + "}", r"\nolinkurl{" + COMPANION_URL + "}",
    r"\href{" + COMPANION_URL + "}{Companion code}", "Code: " + COMPANION_URL + "\n",
])
def test_camera_ready_companion_url_literal(text):
    packager.scan_text(text, "main.tex", camera_ready=True)


@pytest.mark.parametrize("text", [
    COMPANION_URL + suffix for suffix in (
        "/", "/tree/main", ".git", ".evil", "-extra", "?download=1", "#readme",
        "%2Fprivate", "@evil.example", ":443", "\\extra", ";extra", ",extra", "!extra",
    )
] + [
    "https://github.com/magnaprog/conditional-diffusion-consistency",
    "x" + COMPANION_URL, "https://evil.example/?next=" + COMPANION_URL,
    COMPANION_URL.replace("https:", "http:"), COMPANION_URL.replace("github.com", "github.com.evil.example"),
    COMPANION_URL.replace("github.com", "github.com@evil.example"),
    COMPANION_URL.replace("magnaprog", "MAGNAPROG"), COMPANION_URL.upper(),
    COMPANION_URL.replace("/diffusion", "/other"),
    COMPANION_URL.replace("/diffusion", "/\ndiffusion"),
    "\u200b" + COMPANION_URL, COMPANION_URL + "\u200b",
    COMPANION_URL + " magnaprog", COMPANION_URL + " skykev1928",
    COMPANION_URL + " ALISON", COMPANION_URL + " other@example.edu",
    COMPANION_URL + " /home/private/study", COMPANION_URL + " /tmp/private/study",
    COMPANION_URL + r" C:\Users\private\study",
])
def test_camera_ready_companion_url_does_not_exempt_variants_or_other_leaks(text):
    assert text != COMPANION_URL
    with pytest.raises(ValueError, match="author/path leak"):
        packager.scan_text(text, "main.tex", camera_ready=True)


@pytest.mark.parametrize("camera_ready", [False, True])
@pytest.mark.parametrize("suffix", [".zip", ".tar.gz"])
def test_companion_url_bundle_mode(paper, tmp_path, camera_ready, suffix):
    source = paper / "main.tex"
    tex = source.read_text()
    if camera_ready:
        tex = camera_ready_source(tex)
    tex = tex.replace(r"\end{document}", r"\nolinkurl{" + COMPANION_URL + "}\n" + r"\end{document}")
    source.write_text(tex)
    output = tmp_path / ("companion" + suffix)
    if camera_ready:
        assert run(paper, output) == 0
        assert read_archive(output)["main.tex"] == source.read_bytes()
    else:
        assert run(paper, output) != 0
        assert_nothing(output)
