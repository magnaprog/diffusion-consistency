"""Build a minimal arXiv ZIP and verify it against the workshop PDF.

The upload contains only TeX, bibliography files, the required vendor style,
and the eight figures. A fresh output directory holds the ZIP, its source
folder, a preview PDF, and separate verification records. Nothing is submitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import package_prigm_paper as paper_package

FIGURES = {
    "figures/operator/fig_operator.pdf": "figures/figure_01_clipping.pdf",
    "figures/checkpoints/fig_checkpoints.pdf": "figures/figure_02_checkpoint_selection.pdf",
    "figures/hmm/fig_hmm.pdf": "figures/figure_03_exact_denoiser.pdf",
    "figures/calibration/fig_calibration.pdf": "figures/figure_04_scores_and_coverage.pdf",
    "figures/fixed_final/fig_fixed_final.pdf": "figures/figure_05_penalty_strength.pdf",
    "figures/imputation/case_120/fig_imputation.pdf": "figures/figure_06_imputation_window_120.pdf",
    "figures/imputation/case_360/fig_imputation.pdf": "figures/figure_07_imputation_window_360.pdf",
    "figures/imputation/detail/fig_imputation.pdf": "figures/figure_08_variance_predictions.pdf",
}
GRAPHIC = re.compile(r"(\\includegraphics\s*\[[^\]]*\]\s*\{)([^{}]+)(\})")
ZIP_NAME = "residual_penalties_algebraic_completion_arxiv.zip"
PDF_NAME = "residual_penalties_algebraic_completion.pdf"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare_source(files):
    """Rename exactly the eight figure references, preserving all other bytes."""
    source = files["main.tex"].decode("utf-8")
    matches = list(GRAPHIC.finditer(source))
    if [m[2] for m in matches] != list(FIGURES):
        raise ValueError("figure references differ from the eight reviewed figures")
    rewritten = GRAPHIC.sub(lambda m: m[1] + FIGURES[m[2]] + m[3], source)
    inverse = {v: k for k, v in FIGURES.items()}
    if GRAPHIC.sub(lambda m: m[1] + inverse[m[2]] + m[3], rewritten) != source:
        raise ValueError("figure renaming changed other manuscript content")
    result = {name: files[name] for name in ("references.bib", paper_package.STYLE)}
    result["main.tex"] = rewritten.encode("utf-8")
    result.update({new: files[old] for old, new in FIGURES.items()})
    return result


def write_files(directory, files):
    directory.mkdir()
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def check_fonts(report):
    """Require embedded outline fonts, as specified by the NeurIPS template."""
    lines = report.splitlines()
    if len(lines) < 3:
        raise ValueError("pdffonts found no fonts")
    kind = lines[0].index("type")
    encoding = lines[0].index("encoding")
    embedded = lines[0].index("emb")
    allowed = {"Type 1", "Type 1C", "CID Type 0", "CID Type 0C", "TrueType", "CID TrueType"}
    for line in lines[2:]:
        if (line[kind:encoding].strip() not in allowed
                or line[embedded:embedded + 3] != "yes"):
            raise ValueError("PDF requires embedded outline fonts; check pdffonts and CM-Super installation")


def build(output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    commands = []
    environment = os.environ.copy()
    # A compile must not find missing upload files through a repository override.
    for name in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXMFOUTPUT"):
        environment.pop(name, None)

    def run(label, argv, cwd):
        log = logs / (label + ".log")
        started = datetime.now(timezone.utc).isoformat()
        clock = time.monotonic()
        with log.open("xb") as handle:
            result = subprocess.run(argv, cwd=cwd, env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT)
        commands.append({"label": label, "argv": [str(x) for x in argv],
                         "cwd": str(cwd), "started_utc": started,
                         "elapsed_seconds": time.monotonic() - clock,
                         "log": "logs/" + log.name, "returncode": result.returncode,
                         "sha256": sha(log.read_bytes())})
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {log}")
        return log.read_text(errors="replace")

    def compile_tex(directory, prefix, bibliography):
        argv = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                "-halt-on-error", "-file-line-error", "-recorder", "main.tex"]
        run(prefix + "-latex-1", argv, directory)
        if bibliography:
            run(prefix + "-bibtex", ["bibtex", "main"], directory)
        run(prefix + "-latex-2", argv, directory)
        run(prefix + "-latex-3", argv, directory)
        final_log = (directory / "main.log").read_text(errors="replace")
        if re.search(r"Overfull \\[hv]box|Float too large|(?:Reference|Citation).*undefined|"
                     r"There were undefined|Rerun to get|Label\(s\) may have changed", final_log):
            raise ValueError(f"unsettled references or layout in {prefix}; inspect its final log")
        check_fonts(run(prefix + "-fonts", ["pdffonts", "main.pdf"], directory))

    def render(pdf, directory, label):
        directory.mkdir()
        info = run(label + "-pdfinfo", ["pdfinfo", str(pdf)], directory)
        match = re.search(r"^Pages:\s+(\d+)$", info, re.M)
        if not match:
            raise ValueError(f"cannot read page count: {pdf}")
        run(label + "-render", ["pdftoppm", "-r", "100", str(pdf),
                                str(directory / "page")], directory)
        pages = sorted(directory.glob("page-*.ppm"))
        if len(pages) != int(match[1]):
            raise ValueError(f"incomplete rendering: {pdf}")
        return {p.name: sha(p.read_bytes()) for p in pages}

    root = paper_package.DEFAULT_PAPER
    record = {"status": "incomplete", "commands": commands,
              "tex_environment": {key: environment[key] for key in
                                  ("TEXMFHOME", "TEXMFCONFIG", "TEXMFVAR")
                                  if key in environment}}
    try:
        files, stamps, evidence = paper_package.collect(root)
        reference_pdf, reference_stamp = paper_package.read_source(root / "main.pdf")
        run("scientific-verification", [sys.executable, str(paper_package.VERIFIER)], REPO)
        if paper_package.collect(root) != (files, stamps, evidence):
            raise ValueError("workshop source changed during verification")
        upload = prepare_source(files)
        run("pdflatex-version", ["pdflatex", "--version"], REPO)
        run("bibtex-version", ["bibtex", "--version"], REPO)
        with tempfile.TemporaryDirectory(prefix=".prigm-arxiv-", dir=output.parent) as temporary:
            work = Path(temporary)
            original_pdf = work / "workshop.pdf"
            original_pdf.write_bytes(reference_pdf)
            first = work / "first-build"
            write_files(first, upload)
            compile_tex(first, "source", bibliography=True)
            upload["main.bbl"] = (first / "main.bbl").read_bytes()
            staged_zip = work / ZIP_NAME
            paper_package.write_archive(staged_zip, "zip", upload)
            paper_package.verify_archive(staged_zip, "zip", upload)
            extracted = work / "extracted"
            extracted.mkdir()
            # All names, types, duplicate checks and bytes were verified above.
            with zipfile.ZipFile(staged_zip) as archive:
                archive.extractall(extracted)
            # Exercise arXiv's documented use of the supplied main.bbl directly.
            compile_tex(extracted, "extracted", bibliography=False)
            reference_pages = render(original_pdf, work / "reference-pages", "reference")
            source_pages = render(first / "main.pdf", work / "source-pages", "source")
            extracted_pages = render(extracted / "main.pdf", work / "extracted-pages", "extracted")
            if source_pages != reference_pages or extracted_pages != reference_pages:
                raise ValueError("the arXiv build differs from the workshop PDF at 100 DPI")
            if paper_package.collect(root) != (files, stamps, evidence):
                raise ValueError("workshop source changed during packaging")
            if paper_package.read_source(root / "main.pdf") != (reference_pdf, reference_stamp):
                raise ValueError("workshop PDF changed during packaging")
            write_files(output / "source", upload)
            shutil.copyfile(extracted / "main.pdf", output / PDF_NAME)
            # The upload ZIP appears only after both clean builds and comparison.
            os.link(staged_zip, output / ZIP_NAME)
            record.update(status="verified locally", main_file="main.tex", compiler="pdflatex",
                          source_sha256=sha(files["main.tex"]),
                          workshop_pdf_sha256=sha(reference_pdf),
                          preview_pdf_sha256=sha((output / PDF_NAME).read_bytes()),
                          archive_sha256=sha((output / ZIP_NAME).read_bytes()),
                          included={n: {"sha256": sha(b), "bytes": len(b)}
                                    for n, b in sorted(upload.items())},
                          figure_names=FIGURES, page_count=len(reference_pages), render_dpi=100,
                          reference_page_sha256=reference_pages,
                          source_page_sha256=source_pages, extracted_page_sha256=extracted_pages,
                          arxiv_server_compilation="Not performed; review its preview during submission.")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    result = build(args.output)
    print(f"{len(result['included'])} upload files; {result['page_count']} matching pages at 100 DPI")


if __name__ == "__main__":
    main()
