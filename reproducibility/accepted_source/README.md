# Source provenance for the accepted paper

The eight files under `prigm/` in this directory preserve the exact source bytes from commit `ae360b283c2170ada51ad1f403c49ea9d81d0e51`. Their SHA256 hashes identify the code recorded for the accepted paper's original calculations. They are historical copies for verification, not modules to import or execute.

The later [documentation revision](../documentation_source/README.md) corrects
comments and docstrings in some active modules. The verifier first authenticates
that revision and resolves it to its immediate baseline, then applies the
mapping below. The eight snapshots and this manifest retain their original
bytes. Here, "reviewed current module" refers to the version before those later
documentation corrections.

`MANIFEST.json` pairs each `execution_sha256` with the `accepted_sha256` of the reviewed module at the repository root. The paper verifier requires the manifest's recorded hash and authenticates both source versions before returning the hash of the original execution version. For modules outside this mapping, the applicable current or documentation-baseline bytes must match the evidence. Any mismatch with those authenticated versions or the pinned manifest fails verification. The historical evidence and its original source hashes remain unchanged.

| Current module | Reviewed change |
| --- | --- |
| `prigm/artifacts.py` | Reject an ensemble whose sampler differs from its frozen split protocol when the caller supplies the split. |
| `prigm/run_score.py` | Supply the split to that sampler check. |
| `prigm/run_secondaries.py` | Supply the split to that sampler check. |
| `prigm/scoring.py` | Reject invalid score dimensions, too few CRPS draws, and invalid hidden-summary shapes; correct the bootstrap docstring. |
| `prigm/run_cycle.py` | Identify the diagnostic as historical and remove its inaccurate no-redraw claim. |
| `prigm/run_deep_diagnostics.py` | Correct the historical coverage and tensor-layout descriptions; remove unused variables. |
| `prigm/run_persistence.py` | Describe the actual deterministic baseline without an unsupported domain claim. |
| `prigm/run_strata.py` | Disclose Gaussian refitting and sampling and describe the tabulation accurately. |

The core review reported bitwise-identical energy scores and CRPS on 240 valid
cases, identical results for three valid flattening shapes, and byte-identical
legacy diagnostic JSON on nine ensembles. The changes deliberately reject some
previously accepted invalid inputs. This finite review does not establish
equivalence for arbitrary inputs or reproduce historical training. Matching the
snapshots authenticates the recorded source identities; it is not independent
proof of historical execution.

This mapping is used only when verifying the paper. New experiments record the hashes of the code they actually execute. The full repository verifier requires these preserved source copies. The TeX bundles contain neither the copies nor that verifier. The repository MIT license also covers these files.