"""Check that the public calibration copy changes only declared path names.

Compare the exported report with its input after accounting for the path-key
replacement. Numeric values, source hashes and scope statements must match;
the per-window arrays and study specification must be copied byte for byte.
The output manifest records both original identities and exported file hashes.
These tests use locally constructed records and do not rerun calibration.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import export_prigm_calibration as export


@pytest.fixture
def inputs(tmp_path):
    original, accepted = tmp_path / "private-study", tmp_path / "private-evidence"
    source = tmp_path / "private-receipt"
    source.mkdir()
    per_window = source / "per_window.npz"
    np.savez_compressed(per_window, D=np.arange(24., dtype=float).reshape(2, 4, 3),
                        ids=np.array(["2005-08-01", "2005-09-01"]))
    per_window_hash = export.sha256(per_window.read_bytes())
    report = {"schema_version": 1, "methods": {"D": {"mean": [[.1, .7, .2]],
              "year_below_minus_above_95": [[-.12], [-.08]]}},
              "source_sha256": {export.PRODUCER: "a" * 64, "prigm/scoring.py": "b" * 64},
              "per_window_sha256": per_window_hash, "scope": "Retrospective fixed-model comparison",
              "input_sha256": {str(original / "artifacts/test/data.npz"): "c" * 64,
                               str(accepted / "real/MANIFEST.json"): "d" * 64}}
    plan = {"schema_version": 1, "new_analysis_sha256": "a" * 64,
            "methods": ["D"], "source_revision": "e" * 40, "restrictions": ["No refitting"]}
    summary, protocol = source / "summary.json", source / "PROTOCOL.json"
    summary.write_bytes(export.json_bytes(report))
    # Noncanonical formatting must also survive the byte-identical protocol copy.
    protocol.write_text(json.dumps(plan, indent=4) + "\n\n")
    return {"summary": summary, "per_window": per_window, "protocol": protocol,
            "output": tmp_path / "public/calibration", "original_study_root": str(original),
            "accepted_evidence_root": str(accepted), "summary_sha256": export.sha256(summary.read_bytes()),
            "per_window_sha256": per_window_hash, "protocol_sha256": export.sha256(protocol.read_bytes())}


def amend_summary(inputs, change):
    report = json.loads(inputs["summary"].read_bytes())
    change(report)
    inputs["summary"].write_bytes(export.json_bytes(report))
    inputs["summary_sha256"] = export.sha256(inputs["summary"].read_bytes())


def test_exact_projection_bytes_provenance_and_manifest_schema(inputs):
    before = {key: inputs[key].read_bytes() for key in ("summary", "per_window", "protocol")}
    manifest = export.export_calibration(**inputs)
    out = inputs["output"]
    assert {p.name for p in out.iterdir()} == {"summary.json", "per_window.npz", "PROTOCOL.json", "MANIFEST.json"}
    public, original = json.loads((out / "summary.json").read_bytes()), json.loads(before["summary"])
    assert public.pop("input_sha256") == {"original-study/artifacts/test/data.npz": "c" * 64,
                                          "accepted-evidence/real/MANIFEST.json": "d" * 64}
    original.pop("input_sha256")
    assert public == original  # Every numeric value, source hash, and scope string is identical.
    assert (out / "per_window.npz").read_bytes() == before["per_window"]
    assert (out / "PROTOCOL.json").read_bytes() == before["protocol"]
    for key, payload in before.items():
        assert inputs[key].read_bytes() == payload
    assert set(manifest) == {"schema_version", "files", "original_summary_sha256", "protocol_sha256",
                             "producer_sha256", "source", "transformation"}
    assert manifest["original_summary_sha256"] == inputs["summary_sha256"]
    assert manifest["protocol_sha256"] == inputs["protocol_sha256"]
    assert manifest["producer_sha256"] == "a" * 64
    assert manifest["source"] == {"path": "export_prigm_calibration.py",
                                   "sha256": export.sha256(Path(export.__file__).read_bytes())}
    assert manifest["transformation"] == export.TRANSFORMATION
    assert json.loads((out / "MANIFEST.json").read_bytes()) == manifest
    for name, digest in manifest["files"].items():
        assert export.sha256((out / name).read_bytes()) == digest
    for name in ("summary.json", "PROTOCOL.json", "MANIFEST.json"):
        assert str(inputs["summary"].parent.parent) not in (out / name).read_text()


@pytest.mark.parametrize("name", ["summary", "per_window", "protocol"])
def test_changed_input_hash_rejected_without_output(inputs, name):
    inputs[name].write_bytes(inputs[name].read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        export.export_calibration(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("suffix", ["-other/file", "/../escape", "/./file", "//file", "/"])
def test_unknown_or_noncanonical_input_path_rejected(inputs, suffix):
    path = inputs["original_study_root"] + suffix
    amend_summary(inputs, lambda d: d["input_sha256"].update({path: "f" * 64}))
    with pytest.raises(ValueError, match="input path"):
        export.export_calibration(**inputs)
    assert not inputs["output"].exists()


def test_destination_collision_and_duplicate_json_key_rejected():
    with pytest.raises(ValueError, match="collision"):
        export.project_inputs({"/one/file": "a" * 64, "/two/file": "b" * 64},
                              (("/one/", "original-study/"), ("/two/", "original-study/")))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        export.read_json(b'{"input_sha256":{},"input_sha256":{}}')


@pytest.mark.parametrize("field", ["scope", "source_sha256"])
def test_host_path_outside_allowed_keys_is_rejected_not_rewritten(inputs, field):
    def change(d):
        if field == "scope":
            d[field] = "Private source " + inputs["original_study_root"]
        else:
            d[field][inputs["original_study_root"] + "/producer.py"] = "a" * 64
    amend_summary(inputs, change)
    with pytest.raises(ValueError, match="host path remains"):
        export.export_calibration(**inputs)
    assert not inputs["output"].exists()


def test_protocol_path_and_producer_mismatch_rejected(inputs):
    plan = json.loads(inputs["protocol"].read_bytes())
    plan["new_analysis_sha256"] = "b" * 64
    inputs["protocol"].write_bytes(export.json_bytes(plan))
    inputs["protocol_sha256"] = export.sha256(inputs["protocol"].read_bytes())
    with pytest.raises(ValueError, match="producer identity"):
        export.export_calibration(**inputs)
    plan["new_analysis_sha256"] = "a" * 64
    plan["private_root"] = inputs["original_study_root"]
    inputs["protocol"].write_bytes(export.json_bytes(plan))
    inputs["protocol_sha256"] = export.sha256(inputs["protocol"].read_bytes())
    with pytest.raises(ValueError, match="host path remains"):
        export.export_calibration(**inputs)
    assert not inputs["output"].exists()


def test_per_window_identity_and_existing_output_rejected(inputs):
    amend_summary(inputs, lambda d: d.update(per_window_sha256="f" * 64))
    with pytest.raises(ValueError, match="per-window identity"):
        export.export_calibration(**inputs)
    inputs["output"].mkdir(parents=True)
    marker = inputs["output"] / "keep"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        export.export_calibration(**inputs)
    assert marker.read_bytes() == b"original"
