"""Check the contents and byte identity of a public repository copy.

Use the packaging fixtures to verify the exact exported file inventory,
recorded hashes and sizes, copied historical source snapshots, and local
documentation links. Files excluded from the public release must remain
absent. The receipt describes copy identity only; these tests do not repeat
the scientific study or establish that a paper bundle compiles.
"""
import importlib.util
import json
from pathlib import Path
import re
import shutil

import pytest

from tests.test_prigm_package import paper, sha

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_prigm_release", REPO / "scripts/export_prigm_release.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture
def source(paper, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for group in exporter.CODE_DIRS:
        (root / group).mkdir()
        for path in (REPO / group).glob("*.py"):
            shutil.copyfile(path, root / group / path.name)
    for name in exporter.ROOT_FILES:
        shutil.copyfile(REPO / name, root / name)
    shutil.copytree(REPO / exporter.ACCEPTED, root / exporter.ACCEPTED,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(REPO / exporter.documentation.ARCHIVE, root / exporter.documentation.ARCHIVE,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(REPO / exporter.documentation.OCT29_ARCHIVE, root / exporter.documentation.OCT29_ARCHIVE)
    (root / "docs").mkdir()
    shutil.copyfile(REPO / "docs/prigm_release.md", root / "docs/prigm_release.md")
    (root / exporter.PAPER).parent.mkdir(parents=True)
    paper.rename(root / exporter.PAPER)
    (root / exporter.PAPER / "main.pdf").write_bytes(b"%PDF-1.4\nsynthetic manuscript\n%%EOF\n")
    shutil.copyfile(REPO / exporter.PAPER / "fig_audit.py", root / exporter.PAPER / "fig_audit.py")
    for name in ("README.md", "AGENTS.md", "docs/private.md", ".env", "prigm/launch_gpu.sh",
                 "scripts/audit_prigm_prose.py",
                 "paper/workshops/prigm/prigm-overleaf.zip", "paper/workshops/prigm/fig_cycle.py",
                 "paper/workshops/prigm/fig_strata.py"):
        (root / name).write_text("Private excluded fixture.\n")
    return root


def test_export_has_exact_receipt_and_working_public_entrypoints(source, tmp_path):
    output = tmp_path / "release"
    receipt = exporter.export_release(source, output)
    assert json.loads((output / "RELEASE_MANIFEST.json").read_text()) == receipt
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == {
        *receipt["files"], "RELEASE_MANIFEST.json"}
    assert "tests/test_prigm_scientific_argument.py" in receipt["files"]
    assert len(list((output / "prigm").glob("*.py"))) == 43
    assert len(list((output / exporter.ACCEPTED / "prigm").glob("*.py"))) == 8
    for name in exporter.documentation.NAMES:
        snapshot = exporter.documentation.ARCHIVE + "/" + name
        assert (output / snapshot).read_bytes() == (REPO / snapshot).read_bytes()
    revision = exporter.documentation.OCT29_ARCHIVE
    revision_files = {f"{revision}/MANIFEST.json", f"{revision}/README.md",
                      *(f"{revision}/{name}" for name in exporter.documentation.OCT29_NAMES)}
    assert len(revision_files) == 5
    assert {name for name in receipt["files"] if name.startswith(revision + "/")} == revision_files
    for name in revision_files:
        assert (output / name).read_bytes() == (REPO / name).read_bytes()
    assert (output / exporter.PAPER / "fig_audit.py").is_file()
    assert (output / exporter.PAPER / "main.pdf").is_file()
    assert (output / "pytest.ini").read_bytes() == (REPO / "pytest.ini").read_bytes()
    for name, item in receipt["files"].items():
        payload = (output / name).read_bytes()
        assert item == {"sha256": sha(payload), "bytes": len(payload)}
        if name not in receipt["generated"]:
            assert payload == (source / name).read_bytes()
    for name in ("README.md", "AGENTS.md", "docs/prigm_release.md", f"{revision}/README.md"):
        path = output / name
        text = path.read_text()
        assert "Private excluded fixture" not in text
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith(("http:", "https:", "#")):
                assert (path.parent / target.split("#")[0]).resolve().is_file(), target
    for name in (".git", ".env", "docs/private.md", "prigm/launch_gpu.sh",
                 "scripts/audit_prigm_prose.py",
                 exporter.PAPER + "/prigm-overleaf.zip", exporter.PAPER + "/fig_cycle.py",
                 exporter.PAPER + "/fig_strata.py"):
        assert not (output / name).exists()
    assert "Copy identity only" in receipt["validation"]


@pytest.mark.parametrize("destination", ["existing", "symlink", "inside"])
def test_release_requires_fresh_external_destination(source, tmp_path, destination):
    output = tmp_path / "release"
    if destination == "existing":
        output.mkdir()
        (output / "sentinel").write_text("preserve")
    elif destination == "symlink":
        output.symlink_to(tmp_path / "missing")
    else:
        output = source / "release"
    with pytest.raises((FileExistsError, ValueError)):
        exporter.export_release(source, output)
    if destination == "existing":
        assert (output / "sentinel").read_text() == "preserve"


@pytest.mark.parametrize("name", ["prigm/scoring.py", "reproducibility/accepted_source/prigm/scoring.py"])
def test_release_cannot_rebind_original_or_reviewed_source(source, tmp_path, name):
    path = source / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source provenance mismatch"):
        exporter.export_release(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("name", [
    "analysis/prigm_tail_calibration.py", "experiments/_paper_tables.py",
    "prigm/model.py", "scripts/package_prigm_arxiv.py", "tests/test_prigm_core.py",
])
def test_release_refuses_missing_required_python_source(source, tmp_path, name):
    (source / name).unlink()
    with pytest.raises(ValueError, match="missing required Python sources: " + re.escape(name)):
        exporter.export_release(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("name", ["analysis/prigm_tail_calibration.py", ".gitignore"])
def test_release_refuses_symlinked_selected_sources(source, tmp_path, name):
    path = source / name
    payload = tmp_path / "payload"
    path.replace(payload)
    path.symlink_to(payload)
    with pytest.raises(ValueError, match="symlink|regular source"):
        exporter.export_release(source, tmp_path / "release")


@pytest.mark.parametrize("change", ["payload", "missing"])
def test_changed_source_leaves_no_completion_receipt(source, tmp_path, monkeypatch, change):
    original = exporter.collect

    def changed(root):
        collected = original(root)
        if change == "payload":
            (root / "requirements.txt").write_text("changed\n")
        else:
            (root / "analysis/prigm_tail_calibration.py").unlink()
        return collected
    monkeypatch.setattr(exporter, "collect", changed)
    output = tmp_path / "release"
    with pytest.raises((ValueError, FileNotFoundError),
                       match="source.*changed|prigm_tail_calibration"):
        exporter.export_release(source, output)
    assert not (output / "RELEASE_MANIFEST.json").exists()


@pytest.mark.parametrize("change", ["producer", "snapshot", "manifest"])
def test_provenance_edit_during_collection_cannot_complete(source, tmp_path, monkeypatch, change):
    producer = source / "prigm/scoring.py"
    snapshot = source / exporter.ACCEPTED / "prigm/scoring.py"
    manifest = source / exporter.ACCEPTED / "MANIFEST.json"
    trigger = snapshot if change == "snapshot" else producer
    original = exporter.paper_package.read_source
    changed = False

    def mutate_after_read(path):
        nonlocal changed
        result = original(path)
        if path == trigger and not changed:
            changed = True
            if change == "manifest":
                record = json.loads(manifest.read_bytes())
                record["files"]["prigm/scoring.py"]["accepted_sha256"] = "0" * 64
                manifest.write_text(json.dumps(record))
            else:
                path.write_bytes(result[0] + b"\n# concurrent source edit\n")
        return result

    monkeypatch.setattr(exporter.paper_package, "read_source", mutate_after_read)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="source.*changed|source provenance mismatch"):
        exporter.export_release(source, output)
    assert changed
    assert not (output / "RELEASE_MANIFEST.json").exists()


@pytest.mark.parametrize("group", exporter.CODE_DIRS)
def test_undeclared_python_file_does_not_enter_release(source, tmp_path, group):
    extra = f"{group}/internal_audit_only.py"
    (source / extra).write_text("# Private review helper, outside the public inventory.\n")
    output = tmp_path / "release"
    receipt = exporter.export_release(source, output)
    assert extra not in receipt["files"]
    assert not (output / extra).exists()


def test_late_undeclared_python_file_does_not_change_release(source, tmp_path, monkeypatch):
    original = exporter.collect
    extra = "analysis/internal_audit_only.py"

    def add_after_collection(root):
        collected = original(root)
        (root / extra).write_text("# Added after collection; still outside the release.\n")
        return collected

    monkeypatch.setattr(exporter, "collect", add_after_collection)
    output = tmp_path / "release"
    receipt = exporter.export_release(source, output)
    assert extra not in receipt["files"]
    assert not (output / extra).exists()


def test_exported_tree_can_export_again_without_editorial_scanner(source, tmp_path):
    first = tmp_path / "first"
    receipt = exporter.export_release(source, first)
    assert not (first / "scripts/audit_prigm_prose.py").exists()
    repeated = exporter.export_release(first, tmp_path / "second")
    assert repeated["files"] == receipt["files"]


@pytest.mark.parametrize("relative", ["MANIFEST.json", "prigm/schedule.py", "prigm/scoring.py", "README.md"])
@pytest.mark.parametrize("damage", ["missing", "symlink"])
def test_oct29_route_files_are_required_regular_sources(source, tmp_path, relative, damage):
    path = source / exporter.documentation.OCT29_ARCHIVE / relative
    if damage == "missing":
        path.unlink()
    else:
        alternate = tmp_path / "alternate"
        path.rename(alternate)
        path.symlink_to(alternate)
    output = tmp_path / "release"
    with pytest.raises((ValueError, OSError)):
        exporter.export_release(source, output)
    assert not output.exists()


@pytest.mark.parametrize("component", ["current", "previous", "manifest"])
def test_collector_validates_captured_oct29_bytes(source, tmp_path, monkeypatch, component):
    name = "prigm/schedule.py"
    relative = name if component == "current" else f"{exporter.documentation.OCT29_ARCHIVE}/" + (
        "MANIFEST.json" if component == "manifest" else name)
    target = source / relative
    original = exporter.paper_package.read_source
    valid = target.read_bytes()
    intercepted = []

    def altered_read(path):
        payload, stamp = original(path)
        if path == target:
            intercepted.append(path)
            return payload + b"\n", stamp
        return payload, stamp

    monkeypatch.setattr(exporter.paper_package, "read_source", altered_read)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="source provenance mismatch|October 29 documentation manifest"):
        exporter.export_release(source, output)
    assert intercepted == [target]
    assert target.read_bytes() == valid
    assert not output.exists()


@pytest.mark.parametrize("relative", ["MANIFEST.json", "prigm/schedule.py", "prigm/scoring.py"])
def test_oct29_route_change_after_collection_cannot_complete(source, tmp_path, monkeypatch, relative):
    target = source / exporter.documentation.OCT29_ARCHIVE / relative
    original = exporter.collect

    def changed(root):
        files, evidence = original(root)
        target.write_bytes(target.read_bytes() + b"\n")
        return files, evidence

    monkeypatch.setattr(exporter, "collect", changed)
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="source changed during release copy"):
        exporter.export_release(source, output)
    assert not (output / "RELEASE_MANIFEST.json").exists()


def test_oct29_export_does_not_discover_extra_snapshots(source, tmp_path):
    extra = source / exporter.documentation.OCT29_ARCHIVE / "prigm/private.py"
    extra.write_text("# Outside the declared three-file revision.\n")
    output = tmp_path / "release"
    receipt = exporter.export_release(source, output)
    assert extra.relative_to(source).as_posix() not in receipt["files"]
    assert not (output / extra.relative_to(source)).exists()
