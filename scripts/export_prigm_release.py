"""Copy the explicitly included PriGM files into a new release directory.

The export manifest records the copied bytes. Scientific verification and
manuscript compilation are separate checks; this command does not publish files.
Create the final export after review fixes the source versions to be released.
Copies made before that point remain candidates for a later release.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import package_prigm_paper as paper_package
from experiments import _prigm_documentation as documentation

CODE_DIRS = ("analysis", "experiments", "prigm", "scripts", "tests")
PUBLIC_PYTHON_FILES = frozenset("""
analysis/__init__.py
analysis/export_prigm_calibration.py
analysis/export_prigm_paper_evidence.py
analysis/export_prigm_sample_excerpt.py
analysis/export_prigm_toy_evidence.py
analysis/prigm_followup_summary.py
analysis/prigm_full_flank_gaussian.py
analysis/prigm_operator_mechanism.py
analysis/prigm_tail_calibration.py
analysis/prigm_training_strata.py
experiments/__init__.py
experiments/_paper_tables.py
experiments/_prigm_camera_ready.py
experiments/_prigm_checkpoint_evidence.py
experiments/_prigm_documentation.py
experiments/_prigm_illustrations.py
experiments/run_prigm_fixed_final_sensitivity.py
experiments/run_prigm_repair.py
experiments/run_prigm_toy_repair.py
experiments/verify_prigm_paper.py
prigm/__init__.py
prigm/arrays.py
prigm/artifacts.py
prigm/barrier.py
prigm/build_inventory.py
prigm/calendar.py
prigm/checkpoint.py
prigm/config.py
prigm/controls.py
prigm/data.py
prigm/hmm_diagnostics.py
prigm/keys.py
prigm/loop.py
prigm/loss.py
prigm/model.py
prigm/normalize.py
prigm/observation.py
prigm/oracle.py
prigm/precision.py
prigm/prepare_arrays.py
prigm/q15.py
prigm/quality.py
prigm/recovery.py
prigm/run_bootstrap.py
prigm/run_controls.py
prigm/run_cuda_restart.py
prigm/run_cycle.py
prigm/run_deep_diagnostics.py
prigm/run_eval.py
prigm/run_fit.py
prigm/run_gradient_probe.py
prigm/run_persistence.py
prigm/run_score.py
prigm/run_secondaries.py
prigm/run_strata.py
prigm/sampler.py
prigm/schedule.py
prigm/scoring.py
prigm/select.py
prigm/toy.py
prigm/toy_eval.py
prigm/toy_windows.py
prigm/train.py
scripts/export_prigm_release.py
scripts/package_prigm_arxiv.py
scripts/package_prigm_paper.py
scripts/prigm_checkpoint_followup.py
scripts/replay_prigm_ensembles.py
tests/__init__.py
tests/test_prigm_aggregation.py
tests/test_prigm_artifacts.py
tests/test_prigm_arxiv.py
tests/test_prigm_calibration_export.py
tests/test_prigm_calibration_figure.py
tests/test_prigm_camera_ready.py
tests/test_prigm_checkpoint_evidence.py
tests/test_prigm_checkpoint_followup.py
tests/test_prigm_comparison.py
tests/test_prigm_coordinator.py
tests/test_prigm_core.py
tests/test_prigm_cuda_sampling.py
tests/test_prigm_data.py
tests/test_prigm_deadline_paper.py
tests/test_prigm_documentation_source.py
tests/test_prigm_figures.py
tests/test_prigm_fixed_final_figure.py
tests/test_prigm_fixed_final_sensitivity.py
tests/test_prigm_followup_summary.py
tests/test_prigm_full_flank_gaussian.py
tests/test_prigm_gradient_probe.py
tests/test_prigm_illustration_evidence.py
tests/test_prigm_illustration_publication.py
tests/test_prigm_imputation_figure.py
tests/test_prigm_loop.py
tests/test_prigm_neural.py
tests/test_prigm_operator_figure.py
tests/test_prigm_operator_mechanism.py
tests/test_prigm_oracle.py
tests/test_prigm_package.py
tests/test_prigm_paper.py
tests/test_prigm_paper_evidence.py
tests/test_prigm_recovery.py
tests/test_prigm_release_export.py
tests/test_prigm_replay.py
tests/test_prigm_sample_excerpt.py
tests/test_prigm_statistics.py
tests/test_prigm_tail_calibration.py
tests/test_prigm_toy_evidence.py
tests/test_prigm_toy_repair.py
tests/test_prigm_training_strata.py
""".split())
ROOT_FILES = ("LICENSE", "requirements.txt", "environment.yml", "pytest.ini", ".gitignore")
PAPER = "paper/workshops/prigm"
ACCEPTED = "reproducibility/accepted_source"
README = """# Diffusion consistency

Code and public evidence for **Residual Penalties Versus Algebraic Completion In Conditional Diffusion**, accepted at the 2nd Workshop on Principles of Generative Modeling, NeurIPS 2026.

The study compares ways to impose algebraic constraints on conditional diffusion predictions of magnetic statistics during an hour of withheld measurements. It distinguishes mathematical consistency from predictive accuracy. The [paper](paper/workshops/prigm/main.pdf) describes the methods, results and limitations; its [TeX source](paper/workshops/prigm/main.tex) is included.

The [release guide](docs/prigm_release.md) covers installation, CPU tests, evidence verification, data reconstruction and paper builds. The public evidence includes TEST observations, numerical results, toy arrays, selected Gaussian predictions and inputs for the figures. Neural checkpoints, complete neural prediction sets, TRAIN/VAL arrays and raw OMNI files are not included, so the available checks reproduce only part of the experiments. The original three training seeds and earlier exposure to TEST limit the conclusions as stated in the paper.

For raw-data reconstruction, use the [modified Level-3 OMNI one-minute ASCII product](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/hro_modified_format.txt) from the [NASA data directory](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/). Compare downloaded yearly files with the checksums of the original inputs before reconstruction; upstream revisions can change the resulting arrays.

`RELEASE_MANIFEST.json` records the file hashes and sizes in this export. It establishes copy identity; test results and paper compilation require separate checks. The [accepted-source record](reproducibility/accepted_source/README.md) and [documentation-source record](reproducibility/documentation_source/README.md) explain the historical source copies needed to verify the evidence.

Repository code uses the [MIT license](LICENSE). The NeurIPS style keeps its original attribution. NASA OMNI provenance and acknowledgment instructions remain in the linked source documentation.
"""
AGENTS = """# Working on this release

Read [the release guide](docs/prigm_release.md) before changing code or the paper.
Before changing a producer recorded in frozen evidence, read
[the source provenance record](reproducibility/accepted_source/README.md).
Also retain the later [documentation revisions](reproducibility/documentation_source/README.md).
Preserve historical evidence, original source snapshots, and their hashes.
New experiments need separate output directories and provenance.

After changes, run the relevant tests and scientific verifier. Rebuild the PDF
after manuscript edits. Build final archives only from a reviewed source freeze,
then compile each extracted archive. Existing receipts do not certify later edits.
"""


def code_names(root):
    missing = sorted(name for name in PUBLIC_PYTHON_FILES if not (root / name).exists())
    if missing:
        raise ValueError("missing required Python sources: " + ", ".join(missing))
    return set(PUBLIC_PYTHON_FILES)


def collect(root):
    """Collect the manuscript's declared files, release code and source history.

    Reuse the files selected by the paper packager. Authenticate both source
    mappings and retain the same bytes that those checks examined.
    """
    paper_files, _, evidence = paper_package.collect(root / PAPER)
    files = {f"{PAPER}/{name}": value for name, value in paper_files.items() if name != "LICENSE"}
    codes = code_names(root)
    doc_manifest_name = f"{documentation.ARCHIVE}/MANIFEST.json"
    files[doc_manifest_name], _ = paper_package.read_source(paper_package.inside(root, doc_manifest_name))
    doc_manifest = documentation.manifest_record(files[doc_manifest_name])
    for name, record in doc_manifest["files"].items():
        if name not in codes:
            raise ValueError("documentation snapshot has an undeclared producer")
        baseline_name = f"{documentation.ARCHIVE}/{name}"
        for path in (name, baseline_name):
            files[path], _ = paper_package.read_source(paper_package.inside(root, path))
        documentation.validate_revision(name, files[name], files[baseline_name], record)
    names = codes | set(ROOT_FILES) | {
        "docs/prigm_release.md", f"{PAPER}/main.pdf", f"{PAPER}/fig_audit.py",
        f"{documentation.ARCHIVE}/README.md",
        f"{ACCEPTED}/MANIFEST.json", f"{ACCEPTED}/README.md"}
    manifest_name = f"{ACCEPTED}/MANIFEST.json"
    files[manifest_name], _ = paper_package.read_source(paper_package.inside(root, manifest_name))
    manifest = json.loads(files[manifest_name], object_pairs_hook=paper_package.unique_object)
    if manifest["schema_version"] != 1 or len(manifest["files"]) != 8:
        raise ValueError("release requires the eight original source snapshots")
    for name, record in manifest["files"].items():
        if name not in codes or not name.startswith("prigm/"):
            raise ValueError("snapshot has an undeclared current producer")
        snapshot = f"{ACCEPTED}/{name}"
        names.add(snapshot)
        for path, field in ((name, "accepted_sha256"), (snapshot, "execution_sha256")):
            if path not in files:
                files[path], _ = paper_package.read_source(paper_package.inside(root, path))
            payload = files[path]
            baseline = files[f"{documentation.ARCHIVE}/{path}"] if path in documentation.NAMES else payload
            if paper_package.digest(baseline) != record[field]:
                raise ValueError("source provenance mismatch: " + path)
    for name in sorted(names):
        if name in files:
            continue
        path = root / name if name == ".gitignore" else paper_package.inside(root, name)
        files[name], _ = paper_package.read_source(path)
    if files["LICENSE"] != paper_files["LICENSE"]:
        raise ValueError("paper and release licenses differ")
    if not files[f"{PAPER}/main.pdf"].startswith(b"%PDF-"):
        raise ValueError("compiled manuscript is not a PDF")
    return files, evidence


def export_release(root, output):
    root = Path(root).resolve(strict=True)
    output = Path(output).absolute()
    output = output.parent.resolve(strict=True) / output.name
    if os.path.lexists(output):
        raise FileExistsError("release destination must be fresh")
    if output.is_relative_to(root):
        raise ValueError("release destination must be outside the source tree")
    original_codes = code_names(root)
    files, evidence = collect(root)
    copied = dict(files)
    files.update({"README.md": README.encode(), "AGENTS.md": AGENTS.encode()})
    revision = None
    if (root / ".git").exists():
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  check=True, text=True, capture_output=True).stdout.strip()
    output.mkdir()
    for name, payload in sorted(files.items()):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
        if path.read_bytes() != payload:
            raise ValueError("release copy differs: " + name)
    for name, payload in copied.items():
        path = root / name if name == ".gitignore" else paper_package.inside(root, name)
        if paper_package.read_source(path)[0] != payload:
            raise ValueError("source changed during release copy: " + name)
    if (code_names(root) != original_codes
            or paper_package.evidence_names(root / PAPER) != evidence):
        raise ValueError("source inventory changed during release copy")
    manifest = {
        "schema_version": 1, "base_commit": revision,
        "source_state": "Current file bytes, including uncommitted changes; hashes define this export.",
        "validation": "Copy identity only. Scientific tests, PDF verification and compilation are separate checks.",
        "generated": ["AGENTS.md", "README.md"],
        "files": {name: {"sha256": paper_package.digest(payload), "bytes": len(payload)}
                  for name, payload in sorted(files.items())},
        "excluded": ["Git history", "private handoffs and execution receipts", "raw data and neural checkpoints",
                     "old anonymous archives", "build/cache files", "unreferenced historical figures",
                     "legacy GPU shell wrapper", "editorial wording scanner"],
    }
    with (output / "RELEASE_MANIFEST.json").open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory outside this checkout.")
    args = parser.parse_args(argv)
    try:
        manifest = export_release(REPO, args.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print("PriGM release export refused:", exc, file=sys.stderr)
        return 1
    print(f"Copied {len(manifest['files'])} public files; RELEASE_MANIFEST.json written last.")
    print("Copy identity checked. Run scientific verification and compile before release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
