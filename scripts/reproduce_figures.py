"""Redraw the nine figures in the paper from the saved experiment results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("overview", "fixed_final", "checkpoints", "hmm", "calibration", "operator",
           "imputation/case_120", "imputation/case_360", "imputation/detail")
# These inputs are named relative to their experiment directories under evidence/.
EVIDENCE_DIRS = {
    ("fig_hmm", "hmm"): "toy", ("fig_hmm", "oracle"): "toy", ("fig_hmm", "diagnostic"): "toy",
    ("fig_calibration", "calibration"): "followup/calibration", ("fig_calibration", "operator"): "deadline",
}


def arguments(manifest, data_root):
    name, inputs = manifest["figure"], manifest["inputs"]
    if name == "fig_checkpoints":
        record = inputs["MANIFEST.json"]
        return ["--evidence-dir", (data_root / record["path"]).parent, "--manifest-sha256", record["sha256"]]
    args = []
    for role, record in inputs.items():
        if name == "fig_operator" and role != "operator":
            continue  # The renderer locates the TEST observations and Gaussian ensemble itself.
        path = data_root / record["path"]
        if (name, role) in EVIDENCE_DIRS:
            path = data_root / "evidence" / EVIDENCE_DIRS[name, role] / record["path"]
        option = "--evidence" if role == "fixed_final" or name == "fig_operator" else "--" + role
        args.extend([option, path, option + "-sha256", record["sha256"]])
    if name == "fig_imputation":
        if manifest.get("case_index") is None:
            args.append("--detail")
        else:
            args.extend(["--case-index", str(manifest["case_index"])])
    return args


def figure_records():
    for data_root in (ROOT, ROOT / "analysis"):
        if (data_root / "evidence").is_dir():
            producers = data_root / "figures"
            return data_root, producers, json.loads((producers / "inputs.json").read_bytes())
    # In the manuscript checkout, each figure has its own manifest.
    paper = ROOT / "paper/workshops/prigm"
    records = {}
    for folder in FIGURES:
        path, = (paper / "figures" / folder).glob("*.manifest.json")
        records[folder] = json.loads(path.read_bytes())
    return paper, paper, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Write the figures to a new directory.")
    parser.add_argument("--figure", type=int, choices=range(1, len(FIGURES) + 1), help="Draw only this figure. The default is to draw all nine.")
    options = parser.parse_args()
    data_root, producers, records = figure_records()
    options.output.mkdir(parents=True, exist_ok=False)
    for number, folder in enumerate(FIGURES, 1):
        if options.figure is not None and options.figure != number:
            continue
        manifest = records[folder]
        script = producers / (manifest["figure"] + ".py")
        subprocess.run([sys.executable, script, *arguments(manifest, data_root), "--output-dir",
                        options.output / f"figure_{number:02d}"], check=True)


if __name__ == "__main__":
    main()
