"""Check the recorded link between documentation revisions and earlier source bytes.

The current files may have different comments and docstrings from the
preserved copies: thirteen recorded on October 3 and 32 added on October 4.
Their approved hashes must resolve through the reviewed documentation map,
while their Python syntax agrees
after docstrings are removed. Editing a manifest to approve new bytes is
not sufficient: the tests require the recorded map itself to remain intact.

Mutations use temporary copies. These checks neither rewrite the historical
snapshots nor authorize adding later revisions to that map.
"""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from experiments import _prigm_documentation as docs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    shutil.copytree(ROOT / docs.ARCHIVE, tmp_path / docs.ARCHIVE)
    shutil.copytree(ROOT / docs.OCT29_ARCHIVE, tmp_path / docs.OCT29_ARCHIVE)
    for name in docs.NAMES | docs.OCT29_NAMES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


@pytest.mark.parametrize("name", sorted(docs.NAMES))
def test_each_current_documentation_revision_resolves_to_preserved_bytes(source, name):
    baseline = (source / docs.ARCHIVE / name).read_bytes()
    assert (source / name).read_bytes() != baseline
    assert docs.source_hash(source, name) == hashlib.sha256(baseline).hexdigest()


@pytest.mark.parametrize("name", ["prigm/loss.py", "analysis/export_prigm_calibration.py"])
@pytest.mark.parametrize("damage", ["current", "snapshot", "manifest", "missing", "symlink"])
def test_modified_or_missing_provenance_is_rejected(source, damage, name):
    path = source / (docs.ARCHIVE + "/" + name if damage == "snapshot" else name)
    if damage == "manifest":
        path = source / docs.ARCHIVE / "MANIFEST.json"
    if damage == "missing":
        path.unlink()
    elif damage == "symlink":
        alternate = source / "alternate.py"
        path.rename(alternate)
        path.symlink_to(alternate)
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((ValueError, OSError)):
        docs.source_hash(source, name)


def test_coherent_unreviewed_manifest_rebinding_is_rejected(source):
    name = "prigm/loss.py"
    path = source / name
    path.write_bytes(path.read_bytes() + b"\n# new unreviewed wording\n")
    manifest = source / docs.ARCHIVE / "MANIFEST.json"
    record = json.loads(manifest.read_text())
    record["files"][name]["after_sha256"] = docs.digest(path.read_bytes())
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest"):
        docs.source_hash(source, name)


@pytest.mark.parametrize("before,after", [
    (b'return 1', b'return 2'),
    (b'return "observed"', b'return "generated"'),
])
def test_self_consistent_hashes_do_not_excuse_changed_executable_syntax(before, after):
    baseline = b'def f():\n    """Original documentation."""\n    ' + before + b'\n'
    current = b'def f():\n    """New documentation."""\n    ' + after + b'\n'
    record = {"before_sha256": docs.digest(baseline), "after_sha256": docs.digest(current)}
    with pytest.raises(ValueError, match="executable syntax"):
        docs.validate_revision("prigm/loss.py", current, baseline, record)


def test_comments_and_nested_docstrings_can_change_without_losing_code():
    baseline = b'"""Module."""\nclass C:\n    """Class."""\n    def f(self):\n        """Method."""\n        return "result"\n'
    current = baseline.replace(b'Module.', b'Module description.').replace(b'Class.', b'Class description.').replace(b'Method.', b'Method description.') + b'# Note\n'
    record = {"before_sha256": docs.digest(baseline), "after_sha256": docs.digest(current)}
    assert docs.validate_revision("prigm/loss.py", current, baseline, record) == docs.digest(baseline)


@pytest.mark.parametrize("name", ["../outside.py", "/tmp/outside.py"])
def test_source_hash_cannot_read_outside_repository(source, name):
    with pytest.raises(ValueError, match="inside the repository"):
        docs.source_hash(source, name)


@pytest.mark.parametrize("name", sorted(docs.OCT29_NAMES))
def test_oct29_revision_resolves_through_the_previous_active_version(source, name):
    previous = (source / docs.OCT29_ARCHIVE / name).read_bytes()
    current = (source / name).read_bytes()
    manifest = docs.oct29_manifest_record((source / docs.OCT29_ARCHIVE / "MANIFEST.json").read_bytes())
    assert current != previous
    assert docs.validate_oct29_revision(name, current, previous, manifest["files"][name]) == docs.digest(previous)
    if name in docs.NAMES:
        older = (source / docs.ARCHIVE / name).read_bytes()
        original = docs.manifest_record((source / docs.ARCHIVE / "MANIFEST.json").read_bytes())
        expected = docs.validate_revision(name, previous, older, original["files"][name])
        with pytest.raises(ValueError, match="source provenance mismatch"):
            docs.validate_revision(name, current, older, original["files"][name])
    else:
        oracle = json.loads((ROOT / "paper/workshops/prigm/evidence/toy/oracle-stationary-01.json").read_bytes())
        expected = oracle["source_hashes"]["schedule.py"]
        assert docs.digest(previous) == expected
    assert docs.source_hash(source, name) == expected


@pytest.mark.parametrize("name", sorted(docs.OCT29_NAMES))
@pytest.mark.parametrize("component", ["current", "previous", "manifest"])
@pytest.mark.parametrize("damage", ["bytes", "missing", "symlink"])
def test_oct29_route_rejects_changed_missing_or_symlinked_files(source, name, component, damage):
    docs.source_hash(source, name)
    path = source / name if component == "current" else source / docs.OCT29_ARCHIVE / (
        "MANIFEST.json" if component == "manifest" else name)
    if damage == "bytes":
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "missing":
        path.unlink()
    else:
        alternate = source / "alternate"
        path.rename(alternate)
        path.symlink_to(alternate)
    with pytest.raises((ValueError, OSError)):
        docs.source_hash(source, name)


@pytest.mark.parametrize("name", ["analysis/export_prigm_toy_evidence.py", "prigm/scoring.py"])
@pytest.mark.parametrize("damage", ["bytes", "missing", "symlink"])
def test_oct29_chain_still_authenticates_the_older_snapshot(source, name, damage):
    docs.source_hash(source, name)
    path = source / docs.ARCHIVE / name
    if damage == "bytes":
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "missing":
        path.unlink()
    else:
        alternate = source / "older"
        path.rename(alternate)
        path.symlink_to(alternate)
    with pytest.raises((ValueError, OSError)):
        docs.source_hash(source, name)


@pytest.mark.parametrize("name", sorted(docs.OCT29_NAMES))
def test_missing_oct29_route_cannot_fall_back_to_current_hash(source, name):
    docs.source_hash(source, name)
    shutil.rmtree(source / docs.OCT29_ARCHIVE)
    with pytest.raises(OSError):
        docs.source_hash(source, name)


@pytest.mark.parametrize("name", sorted(docs.OCT29_NAMES))
def test_previous_bytes_cannot_replace_the_declared_current_version(source, name):
    docs.source_hash(source, name)
    (source / name).write_bytes((source / docs.OCT29_ARCHIVE / name).read_bytes())
    with pytest.raises(ValueError, match="source provenance mismatch"):
        docs.source_hash(source, name)


def test_oct29_coherent_manifest_rebinding_is_rejected(source):
    name = "prigm/schedule.py"
    docs.source_hash(source, name)
    current = source / name
    current.write_bytes(current.read_bytes() + b"\n# Unreviewed wording.\n")
    manifest = source / docs.OCT29_ARCHIVE / "MANIFEST.json"
    record = json.loads(manifest.read_bytes())
    record["files"][name]["after_sha256"] = docs.digest(current.read_bytes())
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="October 29 documentation manifest"):
        docs.source_hash(source, name)


@pytest.mark.parametrize("change", ["base_commit", "schema_version", "extra_name", "missing_name"])
def test_oct29_inventory_remains_bounded_if_a_pin_is_rebound(monkeypatch, change):
    record = json.loads((ROOT / docs.OCT29_ARCHIVE / "MANIFEST.json").read_bytes())
    if change == "base_commit":
        record["base_commit"] = "0" * 40
    elif change == "schema_version":
        record["schema_version"] = True
    elif change == "extra_name":
        record["files"]["prigm/loss.py"] = dict(record["files"]["prigm/schedule.py"])
    else:
        del record["files"]["prigm/schedule.py"]
    payload = json.dumps(record).encode()
    monkeypatch.setattr(docs, "OCT29_MANIFEST_SHA256", docs.digest(payload))
    with pytest.raises(ValueError, match="inventory differs"):
        docs.oct29_manifest_record(payload)


@pytest.mark.parametrize("name", sorted(docs.OCT29_NAMES))
@pytest.mark.parametrize("expression", [b"2", b'"different"'])
def test_oct29_coherent_pair_hashes_do_not_excuse_executable_changes(name, expression):
    previous = b'def f():\n    """Before."""\n    return 1\n'
    current = b'def f():\n    """After."""\n    return ' + expression + b'\n'
    record = {"before_sha256": docs.digest(previous), "after_sha256": docs.digest(current)}
    with pytest.raises(ValueError, match="executable syntax"):
        docs.validate_oct29_revision(name, current, previous, record)


def test_oct29_pair_validator_does_not_accept_an_unlisted_module():
    previous = b'"""Before."""\nx = 1\n'
    current = b'"""After."""\nx = 1\n'
    record = {"before_sha256": docs.digest(previous), "after_sha256": docs.digest(current)}
    with pytest.raises(ValueError, match="undeclared October 29"):
        docs.validate_oct29_revision("prigm/loss.py", current, previous, record)
    with pytest.raises(ValueError, match="undeclared documentation"):
        docs.validate_revision("prigm/schedule.py", current, previous, record)
