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
    for name in docs.NAMES:
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
