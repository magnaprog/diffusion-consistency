"""Check how reselecting checkpoints changes the O-S minus D score difference on TEST.

Resample calendar years, using the same years for validation and TEST. In each
resample, compare the original checkpoints with those that minimize each fit's
validation score. The calculation runs on CPU using scores saved for the
original fits and ensemble draws. The schema-2 result records the input path
relative to the output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().relative_to(ROOT).as_posix()
CODE = Path("src") if (ROOT / "src/prigm").is_dir() else Path(".")
CALCULATION_SOURCE = (CODE / "scripts/analyze_prigm_checkpoint_reselection.py").as_posix()
CALCULATION_SHA256 = "45c629ad28bf24001c305e1e4a2a2b9db9072af49ab36c684e1757359090d2fa"
HELPERS = tuple((CODE / "prigm" / filename).as_posix()
                for filename in ("scoring.py", "select.py", "config.py"))
EVIDENCE = Path("evidence")
if not (ROOT / EVIDENCE).is_dir():
    EVIDENCE = Path("analysis/evidence")
if not (ROOT / EVIDENCE).is_dir():
    EVIDENCE = Path("paper/workshops/prigm/evidence")
INPUT = (EVIDENCE / "followup/checkpoints/SUMMARY.json").as_posix()


def _load_calculation(payload):
    # Compile the same v1 source bytes whose hash was checked.
    path = ROOT / CALCULATION_SOURCE
    namespace = {"__file__": str(path), "__name__": "_checkpoint_reselection_v1"}
    exec(compile(payload, str(path), "exec"), namespace)
    for name in HELPERS:
        module = sys.modules["prigm." + Path(name).stem]
        if Path(module.__file__).resolve() != (ROOT / name).resolve():
            raise ValueError(f"imported helper differs from the recorded source: {name}")
    return namespace["analyze"]


def write_result(input_path, output_path):
    """Write the results and file hashes to a new JSON file."""
    input_path = Path(input_path).resolve(strict=True)
    output_path = Path(output_path)
    output_path = output_path.parent.resolve(strict=True) / output_path.name
    if input_path == output_path.resolve():
        raise ValueError("output must not overwrite input evidence")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("output already exists; choose a new path")

    sources = {name: (ROOT / name).read_bytes()
               for name in (SOURCE, CALCULATION_SOURCE, *HELPERS)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    if hashes[CALCULATION_SOURCE] != CALCULATION_SHA256:
        raise ValueError("v1 calculation source differs from the frozen version")
    payload = input_path.read_bytes()
    analyze = _load_calculation(sources[CALCULATION_SOURCE])
    result = analyze(json.loads(payload))
    result["schema_version"] = 2
    result["input"] = {
        "path": Path(os.path.relpath(input_path, output_path.parent)).as_posix(),
        "path_base": "output_directory",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    result["source"] = {"path": SOURCE, "sha256": hashes[SOURCE]}
    result["calculation_source"] = {"path": CALCULATION_SOURCE,
                                    "sha256": hashes[CALCULATION_SOURCE]}
    result["helper_sha256"] = {name: hashes[name] for name in HELPERS}
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if input_path.read_bytes() != payload:
        raise ValueError("input changed during analysis")
    for name, data in sources.items():
        if (ROOT / name).read_bytes() != data:
            raise ValueError(f"source changed during analysis: {name}")
    with output_path.open("x") as stream:
        stream.write(serialized)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / INPUT,
                        help="Checkpoint score file. The default is the supplied SUMMARY.json.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Write a new JSON file in an existing directory.")
    args = parser.parse_args()
    try:
        write_result(args.input, args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(args.output)


if __name__ == "__main__":
    main()
