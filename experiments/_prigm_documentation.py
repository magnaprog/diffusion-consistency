"""Authenticate the specifically recorded documentation revisions after the study.

For a listed module, require the current and preserved baseline bytes to
match the two hashes in the reviewed manifest. Then compare their Python
syntax trees after removing docstrings; comments are already absent from
that representation. Return the immediate baseline hash only when both
identity and syntax checks pass. An unlisted module retains its current hash.

This mapping covers the recorded revisions, not arbitrary later wording
changes. It authenticates their relationship to preserved source; it does
not prove that the documentation is scientifically correct. New experiments
continue to record the actual current source bytes through their producers.
"""
import ast
import hashlib
import json
from pathlib import Path


ARCHIVE = "reproducibility/documentation_source"
BASE_COMMIT = "8eff0305d0741ed77ea08be1f33b484aa3e0ed29"
MANIFEST_SHA256 = "34a0d53119f8cae1e0cafa0e2c8e65fb8271ae3f08cace0a7efb878114c4638c"
NAMES = frozenset({
    'analysis/export_prigm_calibration.py',
    'analysis/export_prigm_paper_evidence.py',
    'analysis/export_prigm_sample_excerpt.py',
    'analysis/export_prigm_toy_evidence.py',
    'analysis/prigm_followup_summary.py',
    'analysis/prigm_full_flank_gaussian.py',
    'analysis/prigm_operator_mechanism.py',
    'analysis/prigm_tail_calibration.py',
    'analysis/prigm_training_strata.py',
    'experiments/run_prigm_fixed_final_sensitivity.py',
    'experiments/run_prigm_repair.py',
    'experiments/run_prigm_toy_repair.py',
    'prigm/arrays.py',
    'prigm/artifacts.py',
    'prigm/barrier.py',
    'prigm/build_inventory.py',
    'prigm/checkpoint.py',
    'prigm/controls.py',
    'prigm/data.py',
    'prigm/hmm_diagnostics.py',
    'prigm/loop.py',
    'prigm/loss.py',
    'prigm/model.py',
    'prigm/normalize.py',
    'prigm/observation.py',
    'prigm/oracle.py',
    'prigm/precision.py',
    'prigm/prepare_arrays.py',
    'prigm/q15.py',
    'prigm/recovery.py',
    'prigm/run_bootstrap.py',
    'prigm/run_controls.py',
    'prigm/run_deep_diagnostics.py',
    'prigm/run_eval.py',
    'prigm/run_fit.py',
    'prigm/run_score.py',
    'prigm/run_secondaries.py',
    'prigm/run_strata.py',
    'prigm/sampler.py',
    'prigm/scoring.py',
    'prigm/toy.py',
    'prigm/toy_eval.py',
    'prigm/toy_windows.py',
    'prigm/train.py',
    'scripts/prigm_checkpoint_followup.py',
})


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def manifest_record(payload):
    if digest(payload) != MANIFEST_SHA256:
        raise ValueError("documentation source manifest differs from the reviewed version")
    record = json.loads(payload)
    if (record["schema_version"] != 1 or record["base_commit"] != BASE_COMMIT
            or set(record["files"]) != NAMES):
        raise ValueError("documentation source inventory differs")
    return record


class _WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
        return node


def executable_ast(payload):
    return ast.dump(_WithoutDocstrings().visit(ast.parse(payload)))


def validate_revision(name, current, baseline, record):
    """Check the exact two versions and equality apart from comments/docstrings."""
    if name not in NAMES or set(record) != {"before_sha256", "after_sha256"}:
        raise ValueError("undeclared documentation source revision: " + name)
    if (digest(current) != record["after_sha256"]
            or digest(baseline) != record["before_sha256"]):
        raise ValueError("source provenance mismatch: " + name)
    if executable_ast(current) != executable_ast(baseline):
        raise ValueError("documentation revision changes executable syntax: " + name)
    return record["before_sha256"]


def _read(root, name):
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("documentation source path must stay inside the repository")
    root = Path(root).resolve(strict=True)
    path = root / name
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("documentation source must be a regular file without symlinks: " + name)
    return path.read_bytes()


def source_hash(root, name):
    """Resolve this documentation revision to its exact immediate baseline."""
    current = _read(root, name)
    if name not in NAMES:
        return digest(current)
    record = manifest_record(_read(root, ARCHIVE + "/MANIFEST.json"))
    baseline = _read(root, ARCHIVE + "/" + name)
    return validate_revision(name, current, baseline, record["files"][name])
