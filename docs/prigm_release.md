# PriGM release guide

This tree contains code and public evidence for *Residual Penalties Versus
Algebraic Completion In Conditional Diffusion*, accepted at the 2nd Workshop
on Principles of Generative Modeling, NeurIPS 2026. Read the
[manuscript](../paper/workshops/prigm/main.pdf) for the scientific conclusions
and their limits. Acceptance and a local export do not establish that a
camera-ready revision has been submitted.

`RELEASE_MANIFEST.json` in a copied release directory records every included file's SHA256 and size. The hashes identify the actual copied contents, including any changes made after the recorded Git commit. The manifest checks what was copied; it does not establish that scientific verification or manuscript compilation passed. Recheck edited files and create a new export after further changes.

## Install and check the runtime

Run commands from the repository root. The code imports directly from this
checkout; no editable installation is needed. A CPU installation can use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt pytest
python -m pip check
```

The CPU index and version follow the [official PyTorch 2.6 instructions](https://pytorch.org/get-started/previous-versions/#v260).
On 1 October 2026 this recipe was checked in a fresh Python 3.11.15 environment
without system site packages. It resolved PyTorch 2.6.0+cpu, NumPy 2.4.6,
Matplotlib 3.11.2, and pytest 9.1.1; `pip check` passed. The core, neural CPU,
and then-current tail-analysis tests covered 94 distinct passing cases.
Those dated checks do not certify later source edits or the whole test suite.
The dependency files are not a lock of the original training environment.

The [Conda environment](../environment.yml) specifies Python 3.12 and the same
runtime requirements, with pytest. For GPU work, choose the matching CUDA build
from PyTorch's installation instructions. The legacy shell wrapper is excluded
from the public export because it defaults to a private environment location.
Invoke the Python coordinators directly in your chosen environment.

PDF verification needs Poppler's `pdftotext` and `pdfinfo`, plus `pikepdf`
(included in both installation paths above) to read complete PDF annotation URI
strings. In camera-ready mode, only the exact public companion URI
`https://github.com/magnaprog/diffusion-consistency` is exempt from the
author-identity restriction on links; suffixes, queries and embedded newlines
receive no exception. Anonymous manuscripts, PDF metadata and private paths
retain their restrictions. The catalog URI base is checked as metadata without
this exception, including when no relative links use it. Both raw annotation
URIs and targets resolved against the base are checked; resolution cannot grant
the exception to an annotation whose raw URI differs. Manuscript builds
need pdfLaTeX, BibTeX, and the LaTeX packages loaded by
[main.tex](../paper/workshops/prigm/main.tex), including `orcidlink`. Install
CM-Super with its T1 font mapping to avoid bitmap typewriter fonts. The arXiv
packager also uses Poppler's `pdffonts` and `pdftoppm` to check font embedding
and compare rendered pages.

## Verify this tree

Check the core numerical and neural routines, then verify the public evidence:

```bash
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python -m pytest -q tests/test_prigm_core.py tests/test_prigm_neural.py
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python experiments/verify_prigm_paper.py
```

The verifier compares selected numerical claims and figure records with the public experimental files. It checks which source versions produced those records and examines the supplied manuscript PDF. It does not retrain neural models or certify every sentence. Rebuild the PDF after changing TeX.

For the full CPU test run, choose temporary storage with ample free space.
Some evidence fixtures are large. `pytest.ini` limits default discovery to the
active `tests/` directory. On Linux with sufficient `/dev/shm` capacity:

```bash
if scratch=$(mktemp -d /dev/shm/prigm-tests.XXXXXX); then
  TMPDIR="$scratch" CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg \
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -m pytest -q -rs --basetemp "$scratch/pytest"
fi
```

Two test temporary-directory sites in `tests/test_prigm_operator_mechanism.py`
explicitly use `/dev/shm`; `TMPDIR` does not relocate them. CUDA tests skip in
a CPU-only run, and the real OMNI schema test needs the original 2005 raw file.
Missing PDF tools can cause additional skips. Report the actual results and
skip reasons for the revision being checked.

## Reproduction inputs and limits

The public copy includes TEST observations, toy arrays and results, original prediction scores and other numerical analyses, selected Gaussian predictions, and the inputs used to draw the figures. It also includes the later interval-error and checkpoint records when the manuscript uses them. Neural checkpoints, complete neural prediction sets, raw OMNI data, and TRAIN/VAL arrays are not included. Therefore, the available numerical checks reproduce only part of the experiments.

The [source provenance record](../reproducibility/accepted_source/README.md) explains how eight exact copies of original source files are related to later reviewed code. All 43 current core modules are retained. Preserve the original evidence and its source hashes; results from a new run must remain identifiable as a different execution.

The [documentation revision record](../reproducibility/documentation_source/README.md) maps 45 current files to their versions before the documentation changes: thirteen preserved on October 3 and 32 added on October 4, all from `8eff030`. The initial thirteen copies remain unchanged. The verifier authenticates both versions by hash and compares their Python syntax after removing docstrings. It follows this mapping to the earlier version, then follows the accepted-source mapping to the original execution source where required. Both sets of historical files must retain their bytes. These checks authenticate specific revisions; they do not prove equivalence for every possible runtime observation.

A small hidden Markov model (HMM) execution check uses the included data and a new output file:

```bash
mkdir -p runs
python -m prigm.oracle --toy-eval paper/workshops/prigm/evidence/toy/toy.npz \
  --conditioning pooled --boundary-law stationary --n-windows 1 --n-draws 2 \
  --output runs/oracle-smoke.json
```

This small execution check does not reproduce the paper's estimates. Omit the two size overrides to evaluate the default 288 windows and 39 draws.

For real-data reconstruction, obtain the 2005 through 2024 one-minute ASCII
files in the [modified Level-3 OMNI format](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/hro_modified_format.txt)
from the [NASA data directory](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/).
Place them at `data/omni/omni_min2005.asc` through `omni_min2024.asc` and compare
their hashes with [the public inventory](../paper/workshops/prigm/evidence/data/inventory.json).
Revised upstream files can produce different arrays. Then use a fresh output:

```bash
python -m prigm.prepare_arrays --omni-root data/omni --output runs/inventory
```

The checked data contain 122,309 training windows, 319 validation windows, and 480 test windows. Holding the training windows in memory requires substantial space. On 1 October 2026, the twenty original cached files were independently checked by hash and used to rebuild all seven prepared files, which matched the original files byte for byte. That check required separately stored raw data. A separate examination of NASA's current format and directory listing did not download the yearly data files or establish that their current contents match the original cache.

Training and evaluation entry points are:

```bash
python experiments/run_prigm_repair.py --help
python experiments/run_prigm_toy_repair.py --help
python experiments/run_prigm_fixed_final_sensitivity.py --help
```

Create a separate directory for each new study and retain its declared settings, input hashes and execution records. The coordinators check source identities and enforce the order of validation, model selection and TEST evaluation. Analyses of TEST data that were already examined remain retrospective; generating new predictions does not make those observations an independent test set.

## Build the paper and its source archives

The figure-generating scripts share `paper/workshops/prigm/prigm.mplstyle`. Figure text and mathematics use DejaVu Serif, supplied by Matplotlib. Axis labels and subplot titles are 8 points; ticks, legends and notes are 7 points at the manuscript's display widths. Colors have consistent meanings, and markers, line patterns, hatching and subplot labels also identify the quantities. The Okabe-Ito method palette uses blue for D, vermilion for O-S, bluish green for G5 and reddish purple for Gaussian regression. In Figures 1(b) and 4(a), sky-blue bars show changes in distance to the withheld observations, orange hatched bars show the negative change in dispersion, and purple diamonds show the resulting score change. Figure 1(b) compares predictions after and before clipping; Figure 4(a) compares G5 with D. These score-component colors are distinct from the method palette. The scientific inputs, plotted numerical values, workshop manuscript template and body fonts remain unchanged.

The generators resolve DejaVu Serif without a default-font fallback and embed
TrueType subsets in the PDFs. No separate font installation is needed.
Compiling TeX from the supplied PDFs also needs no plotting-font installation.

After the manuscript is final, rebuild and verify it:

```bash
(
  cd paper/workshops/prigm
  pdflatex -interaction=nonstopmode -halt-on-error main.tex &&
  bibtex main &&
  pdflatex -interaction=nonstopmode -halt-on-error main.tex &&
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
) && python experiments/verify_prigm_paper.py
```

Inspect `paper/workshops/prigm/main.log` after the final LaTeX pass. When it
requests another pass for cross-references, repeat the same `pdflatex` command
from the paper directory and then rerun the verifier. Resolve undefined
references or citations before packaging. The current manuscript has four main
pages and 27 pages in total. Its bibliography uses `unsrtnat` to number entries
in first-citation order. Apply the same checks when compiling each extracted
archive; the final log determines whether another LaTeX pass is needed.

Create fresh archive names; existing archives and sidecars are refused:

```bash
mkdir -p dist
python scripts/package_prigm_paper.py --output dist/prigm-camera-ready-overleaf.zip
python scripts/package_prigm_paper.py --output dist/prigm-camera-ready-source.tar.gz
```

The packaging command runs the verifier and checks the hashes of every file stored in the ZIP or TAR file. Extract each into a new directory and compile it independently. These TeX bundles contain the manuscript dependencies and experimental records, but exclude the full repository verifier, training code and compiled manuscript PDF.

For an arXiv source ZIP, use a fresh output directory:

```bash
python3 scripts/package_prigm_arxiv.py --output dist/prigm-arxiv-YYYYMMDD
```

This command retains the title, author information, workshop layout, bibliography and figure contents. It gives the eight figure PDFs simple filenames and changes only their paths in the exported TeX. The twelve-file ZIP contains `main.tex`, `references.bib`, a newly built `main.bbl`, the original `neurips_2025.sty` and the eight figures. It excludes scientific data, programs and review records.

The packager checks the scientific evidence, compiles both the source and an extraction of the ZIP, and compares every page with the workshop PDF at 100 DPI. It requires embedded outline fonts, settled references and no overfull boxes or oversized figures. Source changes during these checks cause failure. The output directory retains the source files, preview PDF, verification record and logs; failed attempts retain their logs. Use a new directory after resolving a failure.

The ZIP is named `residual_penalties_algebraic_completion_arxiv.zip`, and the preview is `residual_penalties_algebraic_completion.pdf`. The command does not submit the paper or choose its category or license. During submission, select `main.tex` and pdfLaTeX, and inspect the service's generated preview. Local compilation does not establish a successful server build or acceptance.

## Export inventory and maintenance

`scripts/export_prigm_release.py --output /chosen/fresh-directory` copies the public files without Git history and does not publish them. Use a new destination outside the source checkout. The following groups explicitly define which files are included:

| Group | Included material |
| --- | --- |
| Runtime and tooling | The approved Python inventory under `prigm`, `analysis`, `experiments`, `scripts`, and `tests`, recorded in `RELEASE_MANIFEST.json` |
| Source history | The accepted-source manifest, explanation, and eight original source copies; the documentation manifest, explanation, and 45 source copies from before the documentation changes (the initial thirteen plus 32 added on October 4) |
| Paper | Files explicitly required by the current TeX source, public experimental records, figure manifests and generating scripts, and `main.pdf` |
| Figure test support | `fig_audit.py`, even when its figure is not in the manuscript |
| Setup and documentation | License, requirements, environment, `pytest.ini`, ignore rules, this guide, and generated public README/AGENTS |

Private handoffs, execution logs, caches, raw inputs, old anonymous bundles,
unused cycle/strata figures, the editorial wording scanner and the legacy GPU
wrapper are excluded. Historical
document names retained in scientific provenance strings identify original
records; those private planning files are not release instructions.

Repository code uses the [MIT license](../LICENSE). Preserve its copyright
and permission notice when distributing copies. The unchanged NeurIPS style
retains its original attribution. The [NASA source documentation](https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/hro_modified_format.txt)
identifies the data product and requests acknowledgment of SPDF when these
data are used in publications. The repository license does not replace those
source instructions.
