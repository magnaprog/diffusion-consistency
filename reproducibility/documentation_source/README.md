# Source versions before the documentation corrections

The October 3 documentation review preserved thirteen files from commit
`8eff0305d0741ed77ea08be1f33b484aa3e0ed29`. The October 4 prose review adds
32 exact copies from that same commit, giving 45 mapped files. Each added
file matched its active version before the prose edits. The initial thirteen
copies, the eight earlier execution-source copies and all scientific evidence
retain their original bytes.

These are historical source copies, not active modules. The current versions
explain array dimensions, loss weighting, returned values, experiment groups,
score definitions, data processing and the limits of the recorded comparisons.
Output keys, string literals outside docstrings, and numerical expressions
remain unchanged. Edited module docstrings can change command-line help when
a parser uses `__doc__` as its description. The repository MIT license covers
these copies.

`MANIFEST.json` pairs each `before_sha256` with one reviewed `after_sha256`. The verifier authenticates the manifest and both source files by their recorded hashes. It also compares Python syntax trees after removing docstrings; every other statement, expression and literal must agree. This allows only the documented before/after changes. It does not permit an arbitrary replacement source file or prove equivalence for all possible runtime observations. The October 3 manifest is retained in `docs/verification/2026-10-04/full-prose-review/code-integration/documentation-MANIFEST.before.json`; it is a historical review record and is not needed to run the public exporter.

Original inference evidence still refers to the earlier execution sources.
The verifier first resolves the documentation revision to this baseline, then
applies the existing mapping in
[`accepted_source`](../accepted_source/README.md). The October calibration
inventory refers directly to this baseline and uses only the first step.
The eight earlier snapshots and their manifest remain unchanged, as do all
historical scientific records. New runs record current hashes normally.

The public repository exporter includes these versions and retains the exact
bytes that it validates. The TeX archives do not execute the verifier or include
these source copies. Do not import or rewrite the historical modules here.
