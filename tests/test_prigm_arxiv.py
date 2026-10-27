"""Check the arXiv export's manuscript-preservation and file-selection rules."""
from pathlib import Path

import pytest

from scripts import package_prigm_arxiv as arxiv


@pytest.fixture
def source_files():
    root = arxiv.paper_package.DEFAULT_PAPER
    names = ["main.tex", "references.bib", arxiv.paper_package.STYLE, *arxiv.FIGURES]
    return {name: (root / name).read_bytes() for name in names}


def test_only_graphics_paths_change_and_all_other_bytes_survive(source_files):
    result = arxiv.prepare_source(source_files)
    restored = result["main.tex"]
    for old, new in arxiv.FIGURES.items():
        assert result[new] == source_files[old]
        restored = restored.replace(new.encode(), old.encode())
    assert restored == source_files["main.tex"]
    assert result["references.bib"] == source_files["references.bib"]
    assert result[arxiv.paper_package.STYLE] == source_files[arxiv.paper_package.STYLE]


def test_unrelated_files_do_not_enter_upload(source_files):
    source_files.update({"private-notes.md": b"private", "main.pdf": b"compiled",
                         "evidence/run.json": b"{}", "fig_old.pdf": b"unused"})
    result = arxiv.prepare_source(source_files)
    assert set(result) == {"main.tex", "references.bib", arxiv.paper_package.STYLE,
                           *arxiv.FIGURES.values()}
    assert all(Path(name).suffix in {".tex", ".bib", ".sty", ".pdf"} for name in result)


@pytest.mark.parametrize("change", ["unknown", "duplicate", "missing", "order"])
def test_unreviewed_figure_set_requires_explicit_update(source_files, change):
    text = source_files["main.tex"].decode()
    old = next(iter(arxiv.FIGURES))
    if change == "unknown":
        text = text.replace(old, "figures/unreviewed.pdf")
    elif change == "duplicate":
        text += "\n\\includegraphics[width=\\linewidth]{" + old + "}\n"
    elif change == "missing":
        text = arxiv.GRAPHIC.sub("", text, count=1)
    else:
        second = list(arxiv.FIGURES)[1]
        text = text.replace(old, "TEMP").replace(second, old).replace("TEMP", second)
    source_files["main.tex"] = text.encode()
    with pytest.raises(ValueError, match="eight reviewed figures"):
        arxiv.prepare_source(source_files)


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_destination_is_never_replaced(tmp_path, kind):
    path = tmp_path / "existing"
    if kind == "directory":
        path.mkdir()
        (path / "keep.txt").write_text("keep")
    elif kind == "file":
        path.write_text("keep")
    else:
        path.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        arxiv.build(path)
    if kind == "directory":
        assert (path / "keep.txt").read_text() == "keep"
    elif kind == "file":
        assert path.read_text() == "keep"
    else:
        assert path.is_symlink()


@pytest.mark.parametrize("kind,embedded,valid", [
    ("Type 1", "yes", True), ("CID TrueType", "yes", True),
    ("Type 3", "yes", False), ("Type 1", "no", False),
    ("CID TrueType", "no", False),
])
def test_pdf_fonts_require_embedded_outlines(kind, embedded, valid):
    report = (f"{'name':37}{'type':18}{'encoding':17}{'emb':4}sub uni object ID\n"
              + "-" * 90 + "\n"
              + f"{'ExampleFont':37}{kind:18}{'Custom':17}{embedded:4}yes yes 1 0\n")
    if valid:
        arxiv.check_fonts(report)
    else:
        with pytest.raises(ValueError, match="embedded outline"):
            arxiv.check_fonts(report)
