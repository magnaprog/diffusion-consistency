"""Publish a hash-bound calibration copy with only input-path keys changed.

The numeric audit is separate. This helper does not recompute diagnostics.
MANIFEST.json is written last and binds every payload and this exporter.
Private root prefixes are command arguments, never public manifest fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

PRODUCER = "analysis/prigm_tail_calibration.py"
TRANSFORMATION = (
    "Only summary.input_sha256 keys: replace the original-study root prefix with original-study/ "
    "and the accepted-evidence root prefix with accepted-evidence/. All other parsed summary "
    "content is unchanged. per_window.npz and PROTOCOL.json are byte-identical copies."
)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def require_digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA256 digest")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError("nonfinite JSON constant")


def read_json(payload):
    return json.loads(payload, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def root_prefix(value):
    root = str(value).rstrip("/")
    if (not root.startswith("/") or root.startswith("//") or "\\" in root
            or root != str(PurePosixPath(root)) or ".." in PurePosixPath(root).parts
            or any(ord(c) < 32 for c in root)):
        raise ValueError("root must be a canonical absolute POSIX path")
    return root + "/"


def project_inputs(hashes, prefixes):
    projected = {}
    for path, digest in hashes.items():
        require_digest(digest)
        matches = [(private, public) for private, public in prefixes if path.startswith(private)]
        if len(matches) != 1:
            raise ValueError("input path has unknown or ambiguous root")
        private, public = matches[0]
        suffix = path[len(private):]
        if (any(part in ("", ".", "..") for part in suffix.split("/"))
                or "\\" in suffix or any(ord(c) < 32 for c in suffix)):
            raise ValueError("input path suffix must be canonical and nonempty")
        destination = public + suffix
        if destination in projected:
            raise ValueError("public input path collision")
        projected[destination] = digest
    return projected


def reject_host_paths(value):
    """Reject remaining host paths without changing any scientific strings."""
    if isinstance(value, dict):
        for key, child in value.items():
            reject_host_paths(key)
            reject_host_paths(child)
    elif isinstance(value, list):
        for child in value:
            reject_host_paths(child)
    elif isinstance(value, str) and re.search(r"(?<![\w./])(?:/|~/|[A-Za-z]:[\\/])", value):
        raise ValueError("host path remains outside the permitted key transformation")


def export_calibration(summary, per_window, protocol, output, *, summary_sha256,
                       per_window_sha256, protocol_sha256, original_study_root,
                       accepted_evidence_root):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("public calibration output must be fresh")
    paths = {"summary.json": Path(summary), "per_window.npz": Path(per_window),
             "PROTOCOL.json": Path(protocol)}
    expected = dict(zip(paths, (summary_sha256, per_window_sha256, protocol_sha256)))
    payloads = {}
    for name, path in paths.items():
        require_digest(expected[name])
        payloads[name] = path.read_bytes()
        if sha256(payloads[name]) != expected[name]:
            raise ValueError(f"input SHA256 mismatch: {name}")
    report = read_json(payloads["summary.json"])
    plan = read_json(payloads["PROTOCOL.json"])
    if report.get("schema_version") != 1 or plan.get("schema_version") != 1:
        raise ValueError("requires calibration schema version 1")
    if report["per_window_sha256"] != per_window_sha256:
        raise ValueError("summary per-window identity differs")
    producer = require_digest(report["source_sha256"][PRODUCER])
    if plan["new_analysis_sha256"] != producer:
        raise ValueError("protocol producer identity differs from summary")
    prefixes = ((root_prefix(original_study_root), "original-study/"),
                (root_prefix(accepted_evidence_root), "accepted-evidence/"))
    first, second = (p[0] for p in prefixes)
    if first.startswith(second) or second.startswith(first):
        raise ValueError("private root prefixes must be distinct and nonoverlapping")
    public = dict(report)
    public["input_sha256"] = project_inputs(report["input_sha256"], prefixes)
    reject_host_paths(public)
    reject_host_paths(plan)
    payloads["summary.json"] = json_bytes(public)
    source = Path(__file__).resolve()
    source_hash = sha256(source.read_bytes())
    manifest = {
        "schema_version": 1,
        "files": {name: sha256(payload) for name, payload in payloads.items()},
        "original_summary_sha256": summary_sha256,
        "protocol_sha256": protocol_sha256,
        "producer_sha256": producer,
        "source": {"path": source.name, "sha256": source_hash},
        "transformation": TRANSFORMATION,
    }
    output.mkdir(parents=True)
    for name, payload in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(payload)
    if (any(sha256(path.read_bytes()) != expected[name] for name, path in paths.items())
            or sha256(source.read_bytes()) != source_hash):
        raise ValueError("export source or input changed before manifest publication")
    with (output / "MANIFEST.json").open("xb") as handle:
        handle.write(json_bytes(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "per-window", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--original-study-root", required=True)
    parser.add_argument("--accepted-evidence-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    export_calibration(args.summary, args.per_window, args.protocol, args.output_dir,
                       summary_sha256=args.summary_sha256, per_window_sha256=args.per_window_sha256,
                       protocol_sha256=args.protocol_sha256, original_study_root=args.original_study_root,
                       accepted_evidence_root=args.accepted_evidence_root)
    print("Public calibration MANIFEST.json SHA256:", sha256((args.output_dir / "MANIFEST.json").read_bytes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
