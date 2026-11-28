"""Require recorded study settings and selection before follow-up TEST work.

PROTOCOL.json identifies the settings by hash and explicitly records earlier
test exposure. TEST work also requires TEST_STARTED.json to bind the current
SELECTION.json and acknowledge that exposure. These checks preserve the
declared order and identity of follow-up work; they do not make previously
examined test data a new blind evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from prigm.recovery import PublicationError, sha256_file


class TestBarrierError(PublicationError):
    pass


def load_protocol(study_root: Path, expected_hash: str) -> dict:
    path = Path(study_root) / "PROTOCOL.json"
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise TestBarrierError("missing or changed follow-up protocol")
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != 2 or protocol.get("exposure") != "historical_test_exposed":
        raise TestBarrierError("protocol must identify the exposed follow-up")
    return protocol


def require_test_started(study_root: Path, split: str, protocol_hash: str | None = None) -> dict | None:
    split = str(split).upper()
    if split not in {"TRAIN", "VAL", "TEST", "TOY"}:
        raise TestBarrierError(f"unknown split: {split}")
    if split != "TEST":
        return None
    root = Path(study_root)
    phase_path = root / "TEST_STARTED.json"
    selection_path = root / "SELECTION.json"
    if protocol_hash is None:
        raise TestBarrierError("test work requires a pinned protocol hash")
    load_protocol(root, protocol_hash)
    if not phase_path.is_file() or not selection_path.is_file():
        raise TestBarrierError("test work requires a committed selection and exposure receipt")
    phase = json.loads(phase_path.read_text())
    if (phase.get("phase") != "TEST_STARTED"
            or phase.get("protocol_sha256") != protocol_hash
            or phase.get("selection_sha256") != sha256_file(selection_path)
            or phase.get("previous_test_exposure") is not True):
        raise TestBarrierError("test receipt does not bind current selection and prior exposure")
    return json.loads(selection_path.read_text())
