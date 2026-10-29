"""Build a ZIP or TAR file for the anonymous or camera-ready PriGM manuscript.

Include the files named explicitly by the TeX source and the code license.
File names computed by TeX commands are unsupported. Scientific verification
must pass before packaging, but this command does not compile the contents.
Compile each extracted package independently. Neural checkpoints and the
compiled manuscript are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PAPER = REPO / "paper/workshops/prigm"
VERIFIER = REPO / "experiments/verify_prigm_paper.py"
FOLLOWUP_VERIFIERS = {name: REPO / name for name in (
    "experiments/_prigm_camera_ready.py", "experiments/_prigm_checkpoint_evidence.py",
    "experiments/_prigm_documentation.py")}
LICENSE = REPO / "LICENSE"
STYLE = "neurips_2025.sty"
STYLE_SHA256 = "67a960f35da6fecb5a0ce0202fa400459f4c8bb6c294f1a3fdd82fbbd39244c2"
REAL_FILES = {"comparison.json", "diagnostics.json", "scores.json",
              "protocol-summary.json", "gradients.json", "MANIFEST.json"}
DATA_RECORDS = {"quality.json": "quality-01.json", "q15.json": "q15-raw-01.json",
                "inventory.json": "reconstructed_summary.json"}
TOY_FILES = {"oracle-stationary-01.json", "hmm-stationary-01.json", "toy.npz"}
LEARNED_FILES = {"learned.json", "reference-calibration.json", "MANIFEST.json"}
UNCHANGED_RECORDS = {
    "observations": {"test.npz": "artifacts/inventory/test.npz"},
    "sensitivity": {"training-strata.json": "artifacts/training-strata-01.json"},
}
OBSERVATION_FIELDS = {"z", "ids", "split", "mean", "scale", "b_ref_sq", "n_bins"}
GAUSSIAN_FIT_FILES = {"EXTENSION.json", "FIT.npz", "FIT.json", "VALIDATION.json", "SELECTION.json"}
GAUSSIAN_TEST_FILES = {"TEST_STARTED.json", "COMPLETED.json", "SCORES.json"}
GAUSSIAN_CANDIDATES = {(lam, rho): f"lambda_{lam:g}_rho_{rho:g}"
                       for lam in (0.001, 0.1, 1) for rho in (0, 0.5, 1)}
DEADLINE_TRANSFORMATIONS = {
    "operator-mechanism.json": "byte-identical",
    "gaussian": "byte-identical selected evidence; execution locks and LATEST omitted",
    "fixed-final.json": "host paths replaced by original-study/ or extension-study/ prefixes; no numerical changes",
}
OPERATOR_FIGURE = "figures/operator/fig_operator.pdf"
ILLUSTRATION_FIGURES = {
    "figures/fixed_final/fig_fixed_final.pdf",
    "figures/imputation/case_120/fig_imputation.pdf",
    "figures/imputation/case_360/fig_imputation.pdf",
    "figures/imputation/detail/fig_imputation.pdf",
}
IMPUTATION_FILES = {"sample-excerpt.npz", "MANIFEST.json"}
CALIBRATION_FILES = {"MANIFEST.json", "PROTOCOL.json", "summary.json", "per_window.npz"}
CALIBRATION_FIGURE = "figures/calibration/fig_calibration.pdf"
CHECKPOINT_FILES = {"MANIFEST.json", "PROTOCOL.json", "SCOPE.json", "SELECTION.json",
                    "SAVED_VAL25.json", "SUMMARY.json"}
CHECKPOINT_FIGURE = "figures/checkpoints/fig_checkpoints.pdf"
OVERVIEW_FIGURE = "figures/overview/fig_overview.pdf"
RESELECTION_RESULT = "evidence/followup/checkpoint_reselection.json"
RESELECTION_SOURCES = ("scripts/analyze_prigm_checkpoint_reselection.py",
                       "prigm/scoring.py", "prigm/select.py", "prigm/config.py")
CALIBRATION_TRANSFORMATION = (
    "Only summary.input_sha256 keys: replace the original-study root prefix with original-study/ "
    "and the accepted-evidence root prefix with accepted-evidence/. All other parsed summary "
    "content is unchanged. per_window.npz and PROTOCOL.json are byte-identical copies."
)
PACKAGES = {"inputenc", "fontenc", "hyperref", "url", "booktabs", "amsfonts",
            "amsmath", "microtype", "graphicx", "natbib", "times", "lineno", "orcidlink"}
# Additional system packages loaded by supported TeX dependencies.
STYLE_PACKAGES = {"environ", "geometry", "keyval"}
NOTICE_ANONYMOUS = (r"\renewcommand{\@noticestring}{Under review at the 2nd Workshop on "
                    r"Principles of Generative Modeling, NeurIPS 2026.}")
NOTICE_CAMERA_READY = (
    r"\renewcommand{\@noticestring}{40th Conference on Neural Information Processing Systems "
    r"(NeurIPS 2026) Workshop: 2nd Workshop on Principles of Generative Modeling (PriGM).}"
)
NOTICE = NOTICE_ANONYMOUS  # back-compat for anonymous fixtures
STYLE_ANONYMOUS = ("usepackage", "dblblindworkshop", "neurips_2025")
STYLE_CAMERA_READY = ("usepackage", "final,sglblindworkshop", "neurips_2025")
AUTHOR_CAMERA_READY = "Kevin Lee"
ORCID_CAMERA_READY = "0009-0004-0388-9260"
EMAIL_CAMERA_READY = "kevinlee69720@g.ucla.edu"
PUBLIC_COMPANION_URL = "https://github.com/magnaprog/diffusion-consistency"
NATBIB_OPTIONS = r"\PassOptionsToPackage{numbers, sort&compress}{natbib}"
# Private host/identity leaks are refused except the exact camera-ready companion URL.
PRIVATE_LEAK = re.compile(r"magnaprog|skykev|/(?:home|Users|tmp)/|[A-Za-z]:\\Users\\", re.I)
# Author identity refused in anonymous bundles and in non-manuscript members.
AUTHOR_LEAK = re.compile(
    r"\bKevin[\s~{}_,\\-]*Lee\b|\bAlison\b"
    r"|[\w.+-]+@(?:[\w-]+\.)+[a-z]{2,}\b",
    re.I,
)
PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|FIXME|placeholders?)\b|not ready for submission"
    r"|(?:reanalysis|comparison|study|evaluation|fits).*?\bin progress\b"
    r"|\bincomplete study\b|\bresults pending\b", re.I,
)
FILE_COMMAND = re.compile(
    r"\\(includegraphics|input|include|bibliography|bibliographystyle|usepackage|documentclass)\b"
)
ARGUMENT = re.compile(r"\s*(?:\[([^\[\]]*)\]\s*)?\{([^{}]*)\}")
# Do not try to emulate TeX expansion or alternate file search mechanisms.
UNSUPPORTED = re.compile(
    r"\\(?:graphicspath|DeclareGraphicsExtensions|includeonly|import|subimport|"
    r"addbibresource|lstinputlisting|verbatiminput|includepdf|InputIfFileExists|"
    r"IfFileExists|RequirePackage|LoadClass|openin|openout|read|write|write18|"
    r"catcode|csname|def|edef|gdef|xdef|let|futurelet|newcommand|renewcommand|"
    r"providecommand|newenvironment|renewenvironment|endinput|special|"
    r"PassOptionsToPackage|thanks|affil|email|pdfinfo|pdfextension)\b|\\@|\^\^"
)
EXCLUDED = [
    "Every path not listed in included; no paper-root or repository-wide glob.",
    "Unreferenced figures/generators, including historical root fig_*.pdf/png.",
    "Manuscript PDFs, aux/bbl/blg/log/out files, neurips_2026.sty.",
    "Planning documents, execution logs/locks, LATEST, neural checkpoints, credentials, private host paths/metadata, unrelated PISCES assets.",
]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def relative_name(name):
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", name)
            or any(part in ("", ".", "..") for part in name.split("/"))
            or PurePosixPath(name).is_absolute()):
        raise ValueError(f"unsafe or ambiguous relative path: {name!r}")
    if PRIVATE_LEAK.search(name) or AUTHOR_LEAK.search(name):
        raise ValueError(f"author/path leak in member name: {name}")
    return name


def inside(root, name):
    path = root
    for part in relative_name(name).split("/"):
        path /= part
        # Reject internal symlinks too: one literal dependency, one source identity.
        if path.is_symlink():
            raise ValueError(f"symlink dependency refused: {name}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"dependency outside paper root: {name}")
    return path


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def file_id(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def read_source(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular source file: {path.name}")
    payload = path.read_bytes()
    if identity(before) != identity(path.lstat()):
        raise ValueError(f"source changed while reading: {path.name}")
    return payload, identity(before)


def scan_text(text, name, placeholders=False, camera_ready=False):
    if "\x00" in text:
        raise ValueError(f"author/path leak or binary text in {name}")
    private_text = text
    if camera_ready and name == "main.tex":
        # TeX braces and whitespace delimit a complete URL; never exempt a prefix.
        private_text = re.sub(r"(?<![^\s{])" + re.escape(PUBLIC_COMPANION_URL) + r"(?![^\s}])", "", text)
    if PRIVATE_LEAK.search(private_text):
        raise ValueError(f"author/path leak or binary text in {name}")
    if camera_ready and name == "main.tex":
        allowed = {AUTHOR_CAMERA_READY.casefold(), EMAIL_CAMERA_READY.casefold()}
        if any(match[0].casefold() not in allowed for match in AUTHOR_LEAK.finditer(text)):
            raise ValueError(f"author/path leak or binary text in {name}")
    elif AUTHOR_LEAK.search(text):
        raise ValueError(f"author/path leak or binary text in {name}")
    if placeholders and PLACEHOLDER.search(" ".join(text.split())):
        raise ValueError(f"placeholder or incomplete study in {name}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def check_json(payload, name):
    text = payload.decode("utf-8")
    scan_text(text, name, placeholders=True)
    value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)

    def visit(item):
        if isinstance(item, str):
            scan_text(item, name, placeholders=True)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)  # Also scan escaped JSON strings, not just their encoded bytes.
    return value


def check_npz(payload, name):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if not names or len(names) != len(set(names)):
            raise ValueError(f"empty or duplicate NPZ members in {name}")
        if (name == "evidence/observations/test.npz"
                and set(names) != {field + ".npy" for field in OBSERVATION_FIELDS}):
            raise ValueError("observation NPZ field inventory mismatch")
        if (name == "evidence/imputation/sample-excerpt.npz"
                and set(names) != {field + ".npy" for field in ("truth", "samples", "ids", "methods")}):
            raise ValueError("imputation NPZ field inventory mismatch")
        if (name == "evidence/followup/calibration/per_window.npz"
                and set(names) != {field + ".npy" for field in
                                   ("ids", "months", "D", "I-S", "O-S", "G5",
                                    "gaussian", "full_flank_gaussian")}):
            raise ValueError("calibration NPZ field inventory mismatch")
        for member in names:
            relative_name(member)
            if "/" in member or not member.endswith(".npy"):
                raise ValueError(f"unexpected NPZ member in {name}")
            scan_text(member, name)
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        def inspect(array):
            if array.dtype.hasobject:
                raise ValueError(f"object NPZ refused: {name}")
            scan_text(str(array.dtype.descr), name)  # Includes structured field titles.
            if array.dtype.names:
                for field in array.dtype.names:
                    inspect(array[field])
            elif array.dtype.kind in "SU":
                for item in array.flat:
                    text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    scan_text(text, name, placeholders=True)
            elif array.dtype.kind not in "biufcmM":
                raise ValueError(f"unexpected NPZ dtype in {name}")
        for key in arrays.files:
            inspect(arrays[key])  # Object arrays raise here without unpickling.


def gaussian_selected(selection):
    """Return the selected Gaussian directory names from the declared fit grid.

    Require both raw and corrected selections, then check that selected_union
    lists their names in sorted order without duplicates. These names determine
    which nested directories can be included in the public files.
    """
    selected = selection["selected"]
    if not isinstance(selected, dict) or set(selected) != {"raw", "corrected"}:
        raise ValueError("Gaussian selection requires exactly raw/corrected")
    pairs = set()
    for row in selected.values():
        if (not isinstance(row, dict) or set(row) != {"lambda", "rho", "month_balanced_primary_es"}
                or type(row["lambda"]) not in (int, float) or type(row["rho"]) not in (int, float)
                or (row["lambda"], row["rho"]) not in GAUSSIAN_CANDIDATES):
            raise ValueError("Gaussian selection outside fixed parameter grid")
        pairs.add((row["lambda"], row["rho"]))
    names = [GAUSSIAN_CANDIDATES[pair] for pair in sorted(pairs)]
    if not 1 <= len(names) <= 2 or selection["selected_union"] != names:
        raise ValueError("Gaussian selected_union must be sorted, deduplicated selected names")
    return names


def deadline_groups(manifest, selection):
    hash_inventory(manifest["files"])
    fixed = {"fixed-final.json"} if "fixed-final.json" in manifest["files"] else set()
    selected = gaussian_selected(selection)
    groups = {
        "deadline": {"MANIFEST.json", "operator-mechanism.json", "gaussian"} | fixed,
        "deadline/gaussian": GAUSSIAN_FIT_FILES | {"FIT_SELECT.json", "test"},
        "deadline/gaussian/test": GAUSSIAN_TEST_FILES | set(selected),
    }
    for name in selected:
        group = f"deadline/gaussian/test/{name}"
        groups[group] = {"gen_00000000"}
        groups[group + "/gen_00000000"] = {"COMMIT.json", "ensemble.npz"}
    return groups


def evidence_names(root):
    names = []
    groups = {"toy": TOY_FILES | {"learned"}, "toy/learned": LEARNED_FILES,
              "real": REAL_FILES, "data": set(DATA_RECORDS) | {"MANIFEST.json"},
              **{group: set(records) | {"MANIFEST.json"}
                 for group, records in UNCHANGED_RECORDS.items()}}
    if os.path.lexists(root / "evidence/deadline"):
        records = []
        for name in ("MANIFEST.json", "gaussian/SELECTION.json"):
            name = "evidence/deadline/" + name
            payload, _ = read_source(inside(root, name))
            records.append(check_json(payload, name))
        groups.update(deadline_groups(*records))
    if os.path.lexists(root / "evidence/imputation"):
        groups["imputation"] = IMPUTATION_FILES
    if os.path.lexists(root / "evidence/followup"):
        groups["followup"] = {"calibration", "checkpoints"}
        if os.path.lexists(root / RESELECTION_RESULT):
            groups["followup"].add("checkpoint_reselection.json")
        groups["followup/calibration"] = CALIBRATION_FILES
        groups["followup/checkpoints"] = CHECKPOINT_FILES
    directory = inside(root, "evidence")
    if (not directory.is_dir()
            or {p.name for p in directory.iterdir()} != {g.split("/")[0] for g in groups}):
        raise ValueError("unexpected or incomplete evidence directory inventory")
    for group, expected in groups.items():
        directory = inside(root, f"evidence/{group}")
        if not directory.is_dir() or {p.name for p in directory.iterdir()} != expected:
            raise ValueError(f"empty or incomplete evidence/{group}")
        for child in sorted(expected):
            name = f"evidence/{group}/{child}"
            path = inside(root, name)
            if f"{group}/{child}" in groups:
                continue  # Only exact declared nested groups, never arbitrary recursion.
            if not path.is_file() or path.suffix not in (".json", ".npz"):
                raise ValueError(f"unexpected evidence file type: {name}")
            names.append(name)
    return names


def data_bindings(files):
    """Check the original and public hashes recorded for the data files.

    The inventory is an exact copy. Records with private paths shortened have
    different public hashes, while retaining the identities of their originals.
    """
    prefix = "evidence/data/"
    manifest = check_json(files[prefix + "MANIFEST.json"], prefix + "MANIFEST.json")
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or set(manifest["files"]) != set(DATA_RECORDS)
            or set(manifest["records"]) != set(DATA_RECORDS)):
        raise ValueError("data manifest inventory mismatch")
    for name, original in DATA_RECORDS.items():
        spec = manifest["records"][name]
        public = digest(files[prefix + name])
        if (set(spec) != {"source_record", "source_sha256"} or spec["source_record"] != original
                or not isinstance(spec["source_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", spec["source_sha256"])
                or manifest["files"][name] != public
                or (spec["source_sha256"] == public) != (name == "inventory.json")):
            raise ValueError(f"data original/public SHA256 binding mismatch: {name}")


def checked_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected lowercase SHA256")
    return value


def hash_inventory(records):
    if not isinstance(records, dict) or not records:
        raise ValueError("missing SHA256 inventory")
    for name, value in records.items():
        relative_name(name)
        checked_hash(value)


def calibration_bindings(files):
    prefix = "evidence/followup/calibration/"
    if prefix + "MANIFEST.json" not in files:
        return
    records = {n: check_json(files[prefix + n], prefix + n)
               for n in ("MANIFEST.json", "PROTOCOL.json", "summary.json")}
    manifest, plan, summary = (records[n] for n in ("MANIFEST.json", "PROTOCOL.json", "summary.json"))
    if (any(type(r["schema_version"]) is not int or r["schema_version"] != 1
            for r in records.values())
            or set(manifest) != {"schema_version", "files", "original_summary_sha256",
                                 "protocol_sha256", "producer_sha256", "source", "transformation"}
            or manifest["files"] != {n: digest(files[prefix + n])
                                      for n in CALIBRATION_FILES - {"MANIFEST.json"}}
            or manifest["transformation"] != CALIBRATION_TRANSFORMATION):
        raise ValueError("calibration manifest inventory/transformation mismatch")
    checked_hash(manifest["original_summary_sha256"])
    checked_hash(manifest["producer_sha256"])
    if (set(manifest["source"]) != {"path", "sha256"}
            or manifest["source"]["path"] != "export_prigm_calibration.py"):
        raise ValueError("calibration exporter identity mismatch")
    checked_hash(manifest["source"]["sha256"])
    hash_inventory(summary["source_sha256"])
    hash_inventory(summary["input_sha256"])
    if (manifest["protocol_sha256"] != manifest["files"]["PROTOCOL.json"]
            or summary["per_window_sha256"] != manifest["files"]["per_window.npz"]
            or plan["new_analysis_sha256"] != manifest["producer_sha256"]
            or summary["source_sha256"]["analysis/prigm_tail_calibration.py"] != manifest["producer_sha256"]
            or plan["input_manifests"] != {
                group + "/MANIFEST.json": digest(files[f"evidence/{group}/MANIFEST.json"])
                for group in ("real", "deadline")}):
        raise ValueError("calibration protocol/producer/input SHA256 mismatch")


def calibration_figure_bindings(manifest, files):
    """Check the calibration figure's numerical records and input identities.

    Compare tail percentages with the stored analysis and require the declared
    score-component rows. The scientific verifier separately recomputes the
    quantities used to check the figure.
    """
    prefix = "evidence/followup/calibration/"
    summary = check_json(files[prefix + "summary.json"], prefix + "summary.json")
    protocol = check_json(files["evidence/real/protocol-summary.json"], "real protocol")
    required_inputs = {"operator": "evidence/deadline/operator-mechanism.json",
                       "calibration": prefix + "summary.json"}
    if (type(manifest["schema_version"]) is not int
            or set(manifest["inputs"]) != set(required_inputs)
            or manifest["protocol_sha256"] != protocol["source_protocol_sha256"]
            or manifest["channels"] != summary["channels"]
            or manifest["outcomes"] != ["below", "inside", "above"]
            or manifest["method_order"] != ["D", "O-S", "G5", "full_flank_gaussian"]
            or manifest["interval_order_statistics"] != [2, 38]
            or manifest["pointwise_intervals_plotted"] is not False
            or manifest["calibration_provenance"] != {
                "bootstrap": summary["bootstrap"], "per_window_sha256": summary["per_window_sha256"]}):
        raise ValueError("calibration figure schema/provenance mismatch")
    for key, name in required_inputs.items():
        if (manifest["inputs"][key]["path"] not in (name, PurePosixPath(name).name)
                or manifest["inputs"][key]["sha256"] != digest(files[name])):
            raise ValueError("calibration figure input binding mismatch")
    rows = manifest["decomposition"]
    if (not isinstance(rows, list) or len(rows) != 4
            or [r["seed"] for r in rows] != [7, 13, 42, "mean"]
            or any(set(r) != {"seed", "delta_target", "negative_delta_dispersion", "delta_energy"}
                   for r in rows)):
        raise ValueError("calibration decomposition inventory mismatch")
    values = np.asarray([[r[k] for k in ("delta_target", "negative_delta_dispersion", "delta_energy")]
                         for r in rows])
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("calibration decomposition must be finite numeric values")
    if set(manifest["tails"]) != set(manifest["method_order"]):
        raise ValueError("calibration tail method inventory mismatch")
    for method, row in manifest["tails"].items():
        original = summary["methods"][method]
        if row["fixed_model_seeds"] != original["fixed_model_seeds"]:
            raise ValueError("calibration tail model order mismatch")
        for plotted, field in (("mean_percent", "mean"), ("model_means_percent", "model_means"),
                               ("year_95_percent", "year_95")):
            values, expected = np.asarray(row[plotted]), 100 * np.asarray(original[field])
            if (values.dtype.kind not in "iuf" or not np.isfinite(values).all()
                    or not np.array_equal(values, expected)):
                raise ValueError("calibration plotted tail values differ from summary")


def checkpoint_bindings(files):
    prefix = "evidence/followup/checkpoints/"
    if prefix + "MANIFEST.json" not in files:
        return
    records = {n: check_json(files[prefix + n], prefix + n) for n in CHECKPOINT_FILES}
    manifest = records["MANIFEST.json"]
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or set(manifest) != {"schema_version", "analysis", "source_protocol_sha256", "source_scope_sha256",
                                 "source", "files", "original_sha256", "transformations"}
            or manifest["analysis"] != "checkpoint_followup_public_export"
            or manifest["files"] != {n: digest(files[prefix + n]) for n in CHECKPOINT_FILES - {"MANIFEST.json"}}
            or set(manifest["original_sha256"]) != CHECKPOINT_FILES - {"MANIFEST.json"}
            or set(manifest["source"]) != {"path", "sha256"}
            or manifest["source"]["path"] != "fig_checkpoints.py"):
        raise ValueError("checkpoint manifest inventory/source mismatch")
    hash_inventory(manifest["original_sha256"])
    checked_hash(manifest["source"]["sha256"])
    original = manifest["original_sha256"]
    if (manifest["source_protocol_sha256"] != original["PROTOCOL.json"]
            or manifest["source_scope_sha256"] != original["SCOPE.json"]
            or records["PROTOCOL.json"]["scope_sha256"] != original["SCOPE.json"]
            or any(records[n]["protocol_sha256"] != original["PROTOCOL.json"]
                   for n in ("SELECTION.json", "SAVED_VAL25.json", "SUMMARY.json"))
            or any(records[n]["scope_sha256"] != original["SCOPE.json"]
                   for n in ("SELECTION.json", "SAVED_VAL25.json"))
            or records["SUMMARY.json"]["selection_sha256"] != original["SELECTION.json"]
            or original["SAVED_VAL25.json"] not in records["SUMMARY.json"]["input_sha256"].values()):
        raise ValueError("checkpoint original/public provenance mismatch")
    hash_inventory(records["SUMMARY.json"]["input_sha256"])
    transformations = manifest["transformations"]
    if set(transformations) != {"root_replacements", "field_replacements"}:
        raise ValueError("checkpoint transformation inventory mismatch")
    roots, fields = transformations["root_replacements"], transformations["field_replacements"]
    if (len(roots) != 3 or {r["public_root"] for r in roots} != {
            "checkpoint-study", "original-study", "fixed-final-study"}
            or any(set(r) != {"original_root_sha256", "public_root"} for r in roots)
            or len({r["original_root_sha256"] for r in roots}) != 3):
        raise ValueError("checkpoint root transformations mismatch")
    for row in roots:
        checked_hash(row["original_root_sha256"])
    aliases = {"environment.executable": "environment-python", "environment.hostname": "execution-host"}
    if (len(fields) != 2 or {r["field"] for r in fields} != set(aliases)
            or any(set(r) != {"file", "field", "original_value_sha256", "public_value"} for r in fields)):
        raise ValueError("checkpoint field transformations mismatch")
    for row in fields:
        checked_hash(row["original_value_sha256"])
        if (row["file"] != "PROTOCOL.json" or row["public_value"] != aliases[row["field"]]
                or records["PROTOCOL.json"]["environment"][row["field"].split(".")[1]] != row["public_value"]):
            raise ValueError("checkpoint environment alias mismatch")


def checkpoint_reselection_bindings(files, sources):
    """Bind the new sensitivity record to its input and actual source bytes.

    The scientific gate separately recomputes the selection and resampling.
    Historical checkpoint and calibration manifests remain unchanged.
    """
    record = check_json(files[RESELECTION_RESULT], RESELECTION_RESULT)
    expected_fields = {"schema_version", "analysis", "scope", "limits", "procedure",
                       "original_choices", "observed_TEST", "fixed_selection", "reselected",
                       "selection_counts", "input", "source", "helper_sha256"}
    source, *helpers = RESELECTION_SOURCES
    input_name = "evidence/followup/checkpoints/SUMMARY.json"
    if (set(record) != expected_fields or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or record["analysis"] != "checkpoint_reselection_existing_scores"
            or record["input"] != {"path": "paper/workshops/prigm/" + input_name,
                                   "sha256": digest(files[input_name])}
            or record["source"] != {"path": source, "sha256": digest(sources[source])}
            or record["helper_sha256"] != {name: digest(sources[name]) for name in helpers}):
        raise ValueError("checkpoint reselection schema/source/input mismatch")


def checkpoint_figure_bindings(manifest, files):
    prefix = "evidence/followup/checkpoints/"
    provenance = check_json(files[prefix + "MANIFEST.json"], prefix + "MANIFEST.json")
    summary = check_json(files[prefix + "SUMMARY.json"], prefix + "SUMMARY.json")
    saved = check_json(files[prefix + "SAVED_VAL25.json"], prefix + "SAVED_VAL25.json")
    selection = check_json(files[prefix + "SELECTION.json"], prefix + "SELECTION.json")
    if (type(manifest["schema_version"]) is not int
            or manifest.get("evidence_source", manifest["source"]) != provenance["source"]
            or manifest["protocol_sha256"] != provenance["source_protocol_sha256"]
            or len(manifest["inputs"]) != len(CHECKPOINT_FILES)
            or {r["path"] for r in manifest["inputs"].values()} != {prefix + n for n in CHECKPOINT_FILES}
            or manifest["policies"] != selection["policies"]
            or manifest["policies"] != summary["policies"]):
        raise ValueError("checkpoint figure numeric/provenance mismatch")
    expected = {"VAL25": {n: r["means"] for n, r in saved["curves"].items()},
                "VAL50": summary["curves"]["VAL"], "TEST50": summary["curves"]["TEST"]}
    models = {f"{arm}_s{seed}" for arm in ("D", "O-S") for seed in (7, 13, 42)}
    if set(manifest["curves"]) != set(expected):
        raise ValueError("checkpoint figure curve inventory mismatch")
    for split, rows in expected.items():
        if set(rows) != models or set(manifest["curves"][split]) != models:
            raise ValueError("checkpoint figure model inventory mismatch")
        values = np.asarray([manifest["curves"][split][n] for n in sorted(models)])
        original = np.asarray([rows[n] for n in sorted(models)])
        if (values.shape != (6, 5) or original.shape != (6, 5)
                or values.dtype.kind not in "iuf" or original.dtype.kind not in "iuf"
                or not np.isfinite(values).all() or not np.isfinite(original).all()
                or not np.allclose(values, original, rtol=0, atol=1e-12)):
            raise ValueError("checkpoint plotted curves differ from evidence")


def overview_figure_bindings(manifest, files):
    """Compare overview coordinates with the existing public numerical records."""
    prefix = "evidence/followup/checkpoints/"
    summary = check_json(files[prefix + "SUMMARY.json"], "checkpoint summary")
    provenance = check_json(files[prefix + "MANIFEST.json"], "checkpoint provenance")
    calibration_path = "evidence/followup/calibration/summary.json"
    calibration = check_json(files[calibration_path], "calibration summary")
    inputs = {name: prefix + name for name in CHECKPOINT_FILES}
    inputs["calibration"] = calibration_path
    methods = ["D", "O-S", "G5", "full_flank_gaussian"]
    seeds, updates = [7, 13, 42], [2868, 9560, 16252, 22944, 29636]
    if (manifest["inputs"] != {key: {"path": name, "sha256": digest(files[name])}
                               for key, name in inputs.items()}
            or manifest["dependency_sources"] != {
                key: {"path": name, "sha256": digest(files[name])}
                for key, name in (("checkpoints", "fig_checkpoints.py"),
                                  ("calibration", "fig_calibration.py"))}
            or manifest["protocol_sha256"] != provenance["source_protocol_sha256"]
            or manifest["scope_sha256"] != provenance["source_scope_sha256"]
            or manifest["calibration_provenance"] != {
                key: calibration[key] for key in ("bootstrap", "per_window_sha256")}
            or manifest["channels"] != ["mu_x", "mu_y", "mu_z", "m", "v_s"]
            or manifest["channels"] != calibration["channels"]
            or manifest["method_order"] != methods
            or manifest["interval_order_statistics"] != [2, 38]
            or manifest["n_draws"] != 39 or manifest["n_windows"] != 480
            or manifest["coverage_reference"]["percent"] != 90
            or manifest["pointwise_intervals_plotted"] is not False):
        raise ValueError("overview figure input/source/scope mismatch")

    def coordinates(actual, expected):
        values, original = np.asarray(actual), np.asarray(expected)
        if (values.shape != original.shape or values.dtype.kind not in "iuf"
                or original.dtype.kind not in "iuf"
                or not np.isfinite(values).all() or not np.isfinite(original).all()
                or not np.allclose(values, original, rtol=0, atol=1e-12)):
            raise ValueError("overview plotted coordinates differ from evidence")

    a = manifest["panel_a"]
    policies = summary["policies"]
    models = {f"{arm}_s{seed}" for arm in ("D", "O-S") for seed in seeds}
    if (set(a) != {"policy_order", "x", "seeds", "score_by_arm", "seedwise", "mean",
                   "policies", "validation_choices_agree"}
            or a["policy_order"] != ["F", "P25"] or a["seeds"] != seeds
            or a["policies"] != policies or set(policies) != models
            or set(a["score_by_arm"]) != {"D", "O-S"}
            or any(row["P25"] != row["P50"] or row["F"] != updates[-1]
                   for row in policies.values())
            or a["validation_choices_agree"] is not True):
        raise ValueError("overview checkpoint policy/model mismatch")
    coordinates(a["x"], [0, 1])
    scores = {}
    for arm in ("D", "O-S"):
        scores[arm] = np.array([[summary["curves"]["TEST"][f"{arm}_s{seed}"][
            updates.index(policies[f"{arm}_s{seed}"][policy])]
            for policy in ("F", "P25")] for seed in seeds])
        coordinates(a["score_by_arm"][arm], scores[arm])
    differences = scores["O-S"] - scores["D"]
    coordinates(a["seedwise"], differences)
    coordinates(a["mean"], differences.mean(axis=0))
    if set(manifest["panel_b"]) != set(methods):
        raise ValueError("overview coverage method inventory mismatch")
    for index, method in enumerate(methods):
        row, original = manifest["panel_b"][method], calibration["methods"][method]
        expected_seeds = [None] if method == "full_flank_gaussian" else seeds
        if (set(row) != {"fixed_model_seeds", "mean_percent", "model_means_percent", "y", "individual_y"}
                or row["fixed_model_seeds"] != expected_seeds
                or row["fixed_model_seeds"] != original["fixed_model_seeds"]):
            raise ValueError("overview coverage seed inventory mismatch")
        coordinates(row["mean_percent"], 100 * np.asarray(original["mean"])[:, 1])
        coordinates(row["model_means_percent"], 100 * np.asarray(original["model_means"])[:, :, 1])
        positions = np.arange(5) + (index - 1.5) * .15
        coordinates(row["y"], positions)
        coordinates(row["individual_y"], [(positions + (s - 1) * .028).tolist()
                    for s in range(3)] if len(expected_seeds) == 3 else [])


def learned_bindings(files):
    """Check the recorded file relationships for the learned toy results.

    Authenticate the preserved public results and their original input hashes.
    The current exporter source is not substituted for the recorded version.
    """
    prefix = "evidence/toy/learned/"
    records = {n: check_json(files[prefix + n], prefix + n) for n in LEARNED_FILES}
    manifest, learned, reference = (records[n] for n in
                                  ("MANIFEST.json", "learned.json", "reference-calibration.json"))
    if any(type(r["schema_version"]) is not int or r["schema_version"] != 1 for r in records.values()):
        raise ValueError("learned toy schema mismatch")
    if manifest["output_sha256"] != {
            n: digest(files[prefix + n]) for n in LEARNED_FILES - {"MANIFEST.json"}}:
        raise ValueError("learned toy public payload SHA256 mismatch")
    originals, inputs, sources = (manifest[k] for k in ("original_sha256", "input_sha256", "sources"))
    if (set(originals) != {"protocol", "array", "oracle", "trained", "evaluated"}
            or learned["original_sha256"] != originals or learned["sources"] != sources
            or set(sources) != {"training", "inference", "coordinator_sha256"}):
        raise ValueError("learned toy original/source inventory mismatch")
    for inventory in (originals, inputs, sources["training"], sources["inference"]):
        hash_inventory(inventory)
    checked_hash(sources["coordinator_sha256"])
    checked_hash(manifest["exporter_source_sha256"])
    for key, name in (("protocol", "PROTOCOL.json"), ("array", "artifacts/toy.npz"),
                      ("trained", "TRAINED.json"), ("evaluated", "EVALUATED.json")):
        if originals[key] != inputs[name]:
            raise ValueError(f"learned toy original/input SHA256 mismatch: {name}")
    toy_hash = digest(files["evidence/toy/toy.npz"])
    oracle_name = "evidence/toy/oracle-stationary-01.json"
    if (originals["array"] != toy_hash or originals["oracle"] != digest(files[oracle_name])
            or reference["source_oracle_sha256"] != originals["oracle"]
            or reference["source_protocol_sha256"] != originals["protocol"]
            or learned["arrays"] != reference["arrays"] or learned["arrays"]["sha256"] != toy_hash
            or reference["oracle_metadata"]["toy_eval_sha256"] != toy_hash):
        raise ValueError("learned toy public oracle/array/protocol binding mismatch")
    for name in (oracle_name, "evidence/toy/hmm-stationary-01.json"):
        record = check_json(files[name], name)
        if record["toy_eval_sha256"] != toy_hash:
            raise ValueError(f"stationary toy array SHA256 mismatch: {name}")


def unchanged_bindings(files):
    """Check that observation and sensitivity files match the original bytes.

    Require their recorded study and array hashes to agree with the original
    experiment settings as well as with the files included in this package.
    """
    name = "evidence/real/protocol-summary.json"
    protocol = check_json(files[name], name)
    protocol_hash = checked_hash(protocol["source_protocol_sha256"])
    for group, records in UNCHANGED_RECORDS.items():
        prefix = f"evidence/{group}/"
        manifest = check_json(files[prefix + "MANIFEST.json"], prefix + "MANIFEST.json")
        if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
                or set(manifest["files"]) != set(records) or set(manifest["records"]) != set(records)
                or manifest["source_protocol_sha256"] != protocol_hash):
            raise ValueError(f"{group} manifest inventory/protocol mismatch")
        for name, original in records.items():
            public = digest(files[prefix + name])
            if (manifest["files"][name] != public or manifest["records"][name] != {
                    "source_record": original, "source_sha256": public}):
                raise ValueError(f"{group} original/public SHA256 binding mismatch: {name}")
    if digest(files["evidence/observations/test.npz"]) != protocol["arrays"]["TEST"]["sha256"]:
        raise ValueError("observation/protocol TEST SHA256 mismatch")
    name = "evidence/sensitivity/training-strata.json"
    sensitivity = check_json(files[name], name)
    if (type(sensitivity["schema_version"]) is not int or sensitivity["schema_version"] != 1
            or sensitivity["protocol_sha256"] != protocol_hash
            or sensitivity["input_sha256"]["PROTOCOL.json"] != protocol_hash
            or sensitivity["arrays"] != {s: protocol["arrays"][s] for s in ("TRAIN", "TEST")}):
        raise ValueError("sensitivity real protocol/array identity mismatch")


def deadline_bindings(files):
    """Check the complete file set required by the selected Gaussian results.

    Verify file hashes, recorded input relationships and the allowed changes
    made for public export. This check does not recalculate the fitted models.
    """
    prefix = "evidence/deadline/"
    if prefix + "MANIFEST.json" not in files:
        return

    def record(name):
        return check_json(files[prefix + name], prefix + name)

    def hashes(names):
        return {name: digest(files[prefix + name]) for name in names}

    manifest = record("MANIFEST.json")
    selection = record("gaussian/SELECTION.json")
    groups = deadline_groups(manifest, selection)
    expected = {f"{group}/{child}".removeprefix("deadline/")
                for group, children in groups.items() for child in children
                if f"{group}/{child}" not in groups} - {"MANIFEST.json"}
    protocol = check_json(files["evidence/real/protocol-summary.json"], "protocol-summary.json")
    protocol_hash = checked_hash(protocol["source_protocol_sha256"])
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["source_protocol_sha256"] != protocol_hash
            or manifest["files"] != hashes(expected)
            or {n.removeprefix(prefix) for n in files if n.startswith(prefix)} != expected | {"MANIFEST.json"}):
        raise ValueError("deadline manifest inventory/protocol/SHA256 mismatch")
    originals = manifest["original_sha256"]
    required_originals = {"operator-mechanism.json", "gaussian/FIT_SELECT.json"}
    transformations = {k: v for k, v in DEADLINE_TRANSFORMATIONS.items()
                       if k != "fixed-final.json" or k in expected}
    if "fixed-final.json" in expected:
        required_originals.add("fixed-final.json")
    hash_inventory(originals)
    hash_inventory(manifest["producers"])
    if (set(originals) != required_originals or manifest["transformations"] != transformations
            or any(originals[n] != manifest["files"][n]
                   for n in required_originals - {"fixed-final.json"})):
        raise ValueError("deadline original/public SHA256 or transformation mismatch")

    gaussian = "gaussian/"
    fit_hashes = {n: digest(files[prefix + gaussian + n]) for n in GAUSSIAN_FIT_FILES}
    receipt = record(gaussian + "FIT_SELECT.json")
    receipt_hash = manifest["files"][gaussian + "FIT_SELECT.json"]
    if (type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or receipt["phase"] != "FIT_SELECTED" or receipt["source_protocol_sha256"] != protocol_hash
            or receipt["files"] != fit_hashes
            or record(gaussian + "EXTENSION.json")["source_protocol_sha256"] != protocol_hash):
        raise ValueError("Gaussian FIT_SELECT inventory/protocol/SHA256 mismatch")
    for name in ("FIT.json", "VALIDATION.json", "SELECTION.json"):
        metadata = record(gaussian + name)
        if (metadata["extension_protocol_sha256"] != fit_hashes["EXTENSION.json"]
                or metadata["fit_payload_sha256"] != fit_hashes["FIT.npz"]):
            raise ValueError(f"Gaussian fit payload SHA256 mismatch: {name}")
    if (record(gaussian + "FIT.json")["fit_payload"] != "FIT.npz"
            or selection["validation_sha256"] != fit_hashes["VALIDATION.json"]):
        raise ValueError("Gaussian fit/validation binding mismatch")

    selected = gaussian_selected(selection)
    started = record(gaussian + "test/TEST_STARTED.json")
    scores = record(gaussian + "test/SCORES.json")
    completed = record(gaussian + "test/COMPLETED.json")
    for metadata in (started, scores):
        if (metadata["source_protocol_sha256"] != protocol_hash
                or metadata["extension_protocol_sha256"] != fit_hashes["EXTENSION.json"]
                or metadata["fit_select_sha256"] != receipt_hash
                or metadata["fit_payload_sha256"] != fit_hashes["FIT.npz"]
                or metadata["selection_sha256"] != fit_hashes["SELECTION.json"]):
            raise ValueError("Gaussian TEST fit/selection SHA256 mismatch")
    commits = {f"test/{n}/gen_00000000/COMMIT.json" for n in selected}
    completion_files = {"test/TEST_STARTED.json", "test/SCORES.json"} | commits
    if (started["phase"] != "TEST_STARTED" or started["selected_union"] != selected
            or completed["phase"] != "TEST_COMPLETED" or completed["selected_union"] != selected
            or completed["fit_select_sha256"] != receipt_hash
            or completed["files"] != {n: digest(files[prefix + gaussian + n]) for n in completion_files}
            or scores["selected"] != selection["selected"] or set(scores["candidates"]) != set(selected)):
        raise ValueError("Gaussian TEST completion inventory/SHA256 mismatch")
    for name in selected:
        directory = f"test/{name}/gen_00000000/"
        commit = record(gaussian + directory + "COMMIT.json")
        candidate = scores["candidates"][name]
        if (type(commit["generation"]) is not int or commit["generation"] != 0
                or commit["files"] != {"ensemble.npz": digest(files[prefix + gaussian + directory + "ensemble.npz"])}
                or commit["control_fit_sha256"] != fit_hashes["FIT.npz"]
                or commit["fit_select_sha256"] != receipt_hash
                or commit["selection_sha256"] != fit_hashes["SELECTION.json"]
                or candidate["ensemble"] != f"test/{name}"
                or candidate["ensemble_commit_sha256"] != digest(files[prefix + gaussian + directory + "COMMIT.json"])):
            raise ValueError("Gaussian selected COMMIT/ensemble SHA256 mismatch")


def collect(root):
    files, identities = {}, {}

    def add(name):
        relative_name(name)
        if name in files:
            return files[name]
        payload, stamp = read_source(inside(root, name))
        suffix = PurePosixPath(name).suffix
        if name == STYLE:
            if digest(payload) != STYLE_SHA256:
                raise ValueError("canonical vendor style SHA256 mismatch")
            # Preserve the entire canonical vendor file, including its license/authors.
        elif suffix in (".tex", ".bib", ".py", ".mplstyle"):
            decoded = payload.decode("utf-8")
            scan_text(
                decoded,
                name,
                placeholders=suffix == ".tex",
                camera_ready=(name == "main.tex" and NOTICE_CAMERA_READY in decoded),
            )
        elif suffix == ".json":
            check_json(payload, name)
        elif suffix == ".npz":
            check_npz(payload, name)
        elif suffix == ".pdf":
            if not payload.startswith(b"%PDF-"):
                raise ValueError(f"invalid PDF figure: {name}")
        elif suffix == ".png":
            if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"invalid PNG figure: {name}")
        else:
            raise ValueError(f"unexpected file type: {name}")
        files[name], identities[name] = payload, stamp
        return payload

    add(STYLE)
    evidence = evidence_names(root)
    for name in evidence:
        add(name)
    data_bindings(files)
    real_manifest = check_json(files["evidence/real/MANIFEST.json"], "evidence/real/MANIFEST.json")
    if real_manifest["output_sha256"] != {
            n: digest(files["evidence/real/" + n]) for n in REAL_FILES - {"MANIFEST.json"}}:
        raise ValueError("real public payload SHA256 mismatch")
    learned_bindings(files)
    unchanged_bindings(files)
    deadline_bindings(files)
    calibration_bindings(files)
    checkpoint_bindings(files)
    if RESELECTION_RESULT in files:
        checkpoint_reselection_bindings(files, {
            name: read_source(inside(REPO, name))[0] for name in RESELECTION_SOURCES})
    has_followup = "evidence/followup/calibration/MANIFEST.json" in files
    pending, seen, figures, declarations = ["main.tex"], set(), set(), []
    while pending:
        name = pending.pop()
        if name in seen:
            raise ValueError(f"repeated or cyclic TeX input: {name}")
        seen.add(name)
        text = add(name).decode("utf-8")
        text = re.sub(r"(?<!\\)(?:\\\\)*%[^\n]*", "", text)
        if (OPERATOR_FIGURE in text or re.search(r"Full-flank\s+Gaussian|Gaussian\s+regression|tab:fixedfinal", text)):
            if "evidence/deadline/MANIFEST.json" not in files:
                raise ValueError("new deadline claims require evidence/deadline")
        if "tab:fixedfinal" in text and "evidence/deadline/fixed-final.json" not in files:
            raise ValueError("fixed-final claims require evidence/deadline/fixed-final.json")
        if any(label in text for label in ("fig:calibration", "fig:checkpoints", "app:checkpoints", "fig:overview")) and not has_followup:
            raise ValueError("follow-up claims require evidence/followup")
        stripped = (text.replace(NOTICE_ANONYMOUS, "").replace(NOTICE_CAMERA_READY, "")
                         .replace(NATBIB_OPTIONS, ""))
        if UNSUPPORTED.search(stripped):
            raise ValueError(f"unsupported/dynamic TeX command in {name}")
        for match in FILE_COMMAND.finditer(text):
            argument = ARGUMENT.match(text, match.end())
            if argument is None:
                raise ValueError(f"ambiguous TeX reference in {name}")
            options, target = argument.groups()
            command = match.group(1)
            if command in ("input", "include"):
                relative_name(target)
                if not PurePosixPath(target).suffix:
                    if inside(root, target).exists():
                        raise ValueError(f"ambiguous TeX input: {target}")
                    target += ".tex"
                if not target.endswith(".tex") or options is not None:
                    raise ValueError(f"unsupported TeX input: {target}")
                pending.append(target)
            elif command == "includegraphics":
                allowed = {"figures/hmm/fig_hmm.pdf", "figures/audit/fig_audit.pdf", OPERATOR_FIGURE} | ILLUSTRATION_FIGURES
                if has_followup:
                    allowed.update((CALIBRATION_FIGURE, CHECKPOINT_FIGURE, OVERVIEW_FIGURE))
                if target not in allowed:
                    raise ValueError(f"undeclared or ambiguous figure: {target}")
                if options and re.search(r"\b(?:draft|type|ext|read|command)\b", options):
                    raise ValueError("unsupported figure options")
                figures.add(target)
            else:
                declarations.append((command, options, target))
        if name == "main.tex":
            has_anon = text.count(NOTICE_ANONYMOUS) == 1 and STYLE_ANONYMOUS[1] in text
            has_cr = text.count(NOTICE_CAMERA_READY) == 1 and STYLE_CAMERA_READY[1] in text
            if has_anon == has_cr:
                raise ValueError("main.tex must be exactly anonymous or camera-ready")
            if has_anon:
                authors = re.findall(r"\\author\s*\{([^{}]*)\}", text)
                if authors != ["Anonymous"]:
                    raise ValueError("only a literal Anonymous author is permitted")
                if re.search(r"\bpdfauthor\b", re.sub(r"\bpdfauthor\s*=\s*\{\s*\}", "", text)):
                    raise ValueError("PDF author metadata must be explicitly empty")
            else:
                if AUTHOR_CAMERA_READY not in text or ORCID_CAMERA_READY not in text:
                    raise ValueError("camera-ready author/ORCID missing")
                if EMAIL_CAMERA_READY not in text:
                    raise ValueError("camera-ready corresponding email missing")
                if "Alison" in text:
                    raise ValueError("PriGM camera-ready must not list Alison J. March")
                if not re.search(r"\bpdfauthor\s*=\s*\{" + re.escape(AUTHOR_CAMERA_READY) + r"\}", text):
                    raise ValueError("camera-ready pdfauthor must be Kevin Lee")
                if r"\usepackage{orcidlink}" not in text and "orcidlink" not in text:
                    raise ValueError("camera-ready requires orcidlink")
        elif r"\author" in text:
            authors = re.findall(r"\\author\s*\{([^{}]*)\}", text)
            if authors and authors != ["Anonymous"]:
                raise ValueError("only a literal Anonymous author is permitted")
    if has_followup and not {CALIBRATION_FIGURE, CHECKPOINT_FIGURE}.issubset(figures):
        raise ValueError("follow-up figure family is incomplete")
    if figures & ILLUSTRATION_FIGURES or "evidence/imputation/MANIFEST.json" in files:
        if not ILLUSTRATION_FIGURES.issubset(figures):
            raise ValueError("illustration figure family is incomplete")
        if not {"evidence/deadline/fixed-final.json", "evidence/imputation/MANIFEST.json",
                "evidence/imputation/sample-excerpt.npz"}.issubset(files):
            raise ValueError("illustration figures require complete fixed-final and imputation evidence")
        excerpt = "evidence/imputation/sample-excerpt.npz"
        provenance = check_json(files["evidence/imputation/MANIFEST.json"], "imputation provenance")
        if provenance["files"] != {"sample-excerpt.npz": digest(files[excerpt])}:
            raise ValueError("imputation excerpt hash differs")
    main_text = files["main.tex"].decode("utf-8")
    style = STYLE_CAMERA_READY if NOTICE_CAMERA_READY in main_text else STYLE_ANONYMOUS
    required = [("documentclass", None, "article"),
                style,
                ("bibliographystyle", None, "unsrtnat")]
    for declaration in required:
        if declarations.count(declaration) != 1:
            raise ValueError(f"missing/ambiguous template declaration: {declaration[0]}")
    bibs = [d for d in declarations if d[0] == "bibliography"]
    if len(bibs) != 1 or bibs[0][1] is not None or bibs[0][2] not in ("references", "references.bib"):
        raise ValueError("requires only references.bib")
    add("references.bib")
    for declaration in declarations:
        if declaration in required or declaration in bibs:
            continue
        command, options, target = declaration
        if command != "usepackage" or any(p.strip() not in PACKAGES for p in target.split(",")):
            raise ValueError(f"unsupported TeX dependency: {target}")
    # Local shadows of system TeX dependencies would change independent compilation.
    for name in ["article.cls", "unsrtnat.bst", "graphics.cfg",
                 *(p + ".sty" for p in PACKAGES | STYLE_PACKAGES)]:
        if os.path.lexists(root / name):
            raise ValueError(f"local system-TeX shadow refused: {name}")
    for figure in sorted(figures):
        pdf = PurePosixPath(figure)
        stem = pdf.stem
        png = str(pdf.with_suffix(".png"))
        source = stem + ".py"
        for name in (figure, png, source):
            add(name)
        manifest = check_json(add(str(pdf.with_suffix(".manifest.json"))), figure)
        expected_schema = 2 if stem == "fig_hmm" else 1
        if (type(manifest["schema_version"]) is not int
                or manifest["schema_version"] != expected_schema or manifest["figure"] != stem
                or manifest["source"] != {"path": source, "sha256": digest(files[source])}
                or manifest["output_sha256"] != {
                    PurePosixPath(n).name: digest(files[n]) for n in (figure, png)}):
            raise ValueError(f"figure companion/generator SHA256 mismatch: {figure}")
        if b"prigm.mplstyle" in files[source] or "style" in manifest:
            style = add("prigm.mplstyle")
            if manifest.get("style") != {"path": "prigm.mplstyle", "sha256": digest(style)}:
                raise ValueError(f"figure shared-style SHA256 mismatch: {figure}")
        if not isinstance(manifest["inputs"], dict) or not manifest["inputs"]:
            raise ValueError(f"missing figure inputs: {figure}")
        for record in manifest["inputs"].values():
            reference = relative_name(record["path"])
            candidates = [n for n in evidence if n == reference or PurePosixPath(n).name == reference]
            if len(candidates) != 1 or digest(files[candidates[0]]) != record["sha256"]:
                raise ValueError(f"missing/ambiguous or changed figure input: {reference}")
        if figure == CALIBRATION_FIGURE:
            calibration_figure_bindings(manifest, files)
        elif figure == CHECKPOINT_FIGURE:
            checkpoint_figure_bindings(manifest, files)
        elif figure == OVERVIEW_FIGURE:
            for name in ("fig_checkpoints.py", "fig_calibration.py"):
                add(name)
            overview_figure_bindings(manifest, files)
        if "toy_eval_sha256" in manifest and manifest["toy_eval_sha256"] not in {
                digest(files[n]) for n in evidence if n.endswith(".npz")}:
            raise ValueError("figure toy NPZ SHA256 mismatch")
    # Trusted repository notice covers the bundled generators. Preserve its
    # copyright attribution, as with the canonical vendor style above.
    files["LICENSE"], identities["LICENSE"] = read_source(LICENSE)
    return files, identities, evidence


def write_archive(path, kind, files):
    if kind == "zip":
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED)
    else:
        with tarfile.open(path, "x:gz") as archive:
            for name, payload in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(payload), 0o644
                archive.addfile(info, io.BytesIO(payload))


def verify_archive(path, kind, files):
    seen = set()

    def check(name, payload):
        if name in seen or name not in files or digest(payload) != digest(files[name]):
            raise ValueError(f"archive member/hash mismatch: {name}")
        seen.add(name)
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not stat.S_ISREG(member.external_attr >> 16):
                    raise ValueError("non-regular ZIP member")
                check(member.filename, archive.read(member))
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    raise ValueError("non-regular tar member")
                with archive.extractfile(member) as handle:
                    check(member.name, handle.read())
    if seen != set(files):
        raise ValueError("archive member names differ from selected sources")


def package(paper_dir, output):
    root = Path(paper_dir).resolve(strict=True)
    output = Path(output).absolute()
    # Resolve the parent only: a dangling final symlink is an existing output.
    output = output.parent.resolve(strict=True) / output.name
    sidecar = Path(str(output) + ".manifest.json")
    kind = "zip" if output.name.endswith(".zip") else "tar"
    if kind == "tar" and not output.name.endswith(".tar.gz"):
        raise ValueError("output must end with .zip or .tar.gz")

    def fresh():
        for path in (output, sidecar):
            if os.path.lexists(path):
                raise FileExistsError(f"refusing existing output or sidecar: {path.name}")
    fresh()
    root_identity = file_id(root)
    files, stamps, evidence = collect(root)
    verifier_bytes, verifier_stamp = read_source(VERIFIER)
    followup_verifiers = {
        name: read_source(path) for name, path in FOLLOWUP_VERIFIERS.items()
        if name == "experiments/_prigm_documentation.py"
        or "evidence/followup/calibration/MANIFEST.json" in files}

    def unchanged():
        # Rediscover as well as reread: an added extensionless input or system
        # style shadow changes what TeX reads even when selected bytes are intact.
        if file_id(root) != root_identity or collect(root) != (files, stamps, evidence):
            raise ValueError("source changed/replaced during packaging")
        if read_source(VERIFIER) != (verifier_bytes, verifier_stamp):
            raise ValueError("final verifier changed during packaging")
        for name, recorded in followup_verifiers.items():
            if read_source(FOLLOWUP_VERIFIERS[name]) != recorded:
                raise ValueError("follow-up verifier changed during packaging: " + name)

    # No temporary archive or published output exists before this final gate.
    subprocess.run([sys.executable, str(VERIFIER), "--paper-dir", str(root)],
                   cwd=REPO, check=True)
    unchanged()
    fresh()
    with tempfile.TemporaryDirectory(prefix=".prigm-package-", dir=output.parent) as directory:
        temporary = Path(directory) / "archive"
        temporary_manifest = Path(directory) / "manifest.json"
        write_archive(temporary, kind, files)
        verify_archive(temporary, kind, files)
        archive_hash = digest(temporary.read_bytes())
        receipt = {
            "schema_version": 1, "archive_format": kind, "sha256": archive_hash,
            "included": {n: {"sha256": digest(p), "bytes": len(p)} for n, p in sorted(files.items())},
            "excluded": EXCLUDED,
            "verification": {"script": "experiments/verify_prigm_paper.py",
                             "sha256": digest(verifier_bytes), "returncode": 0},
            "compilation": "Not performed. Independently compile each extracted archive.",
        }
        if followup_verifiers:
            receipt["verification"]["supporting_sources"] = {
                name: digest(recorded[0]) for name, recorded in followup_verifiers.items()}
        with temporary_manifest.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
        unchanged()
        published = []
        try:
            fresh()
            # Create the final names as hard links to complete files. If another
            # process already created either name, fail without replacing its file.
            os.link(temporary, output)
            published.append((output, file_id(temporary)))
            verify_archive(output, kind, files)
            if digest(output.read_bytes()) != archive_hash:
                raise ValueError("published archive SHA256 mismatch")
            unchanged()
            os.link(temporary_manifest, sidecar)  # Completion marker is always last.
            published.append((sidecar, file_id(temporary_manifest)))
        except BaseException:
            # Remove only output names that still refer to this attempt's files;
            # preserve a file that another process replaced at the same name.
            for path, inode in reversed(published):
                if os.path.lexists(path) and file_id(path) == inode:
                    path.unlink()
            raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh .zip or .tar.gz; <output>.manifest.json must also be fresh.")
    args = parser.parse_args(argv)
    try:
        receipt = package(args.paper_dir, args.output)
    except subprocess.CalledProcessError as exc:
        return exc.returncode if exc.returncode > 0 else 128 - exc.returncode
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError,
            zipfile.BadZipFile, tarfile.TarError) as exc:
        print(f"PriGM packaging refused: {exc}", file=sys.stderr)
        return 1
    print(f"Packaged {len(receipt['included'])} files; archive roundtrip verified. Compilation not checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
