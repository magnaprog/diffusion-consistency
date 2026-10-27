"""Restart/recovery tests for RecoverableTrainer. CPU only."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prigm.config import N_BINS, TOY_WIDTHS
from prigm.loop import RecoverableTrainer, RunContract, configure_determinism
from prigm.normalize import fit_normalizer
from prigm.recovery import PublicationError
from prigm.toy import pooled_toy_observations


def _data(n=20, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 6, N_BINS)).astype(np.float32)
    obs = u.copy()
    obs[:, :, 18:30] = 0.0
    nrm = fit_normalizer(pooled_toy_observations())
    return u, obs, nrm


def _contract(n, seed=7):
    return RunContract(
        arm="D",
        seed=seed,
        n_generated=6,
        n_aux=0,
        widths=TOY_WIDTHS,
        c_r=0.0,
        residual=None,
        estimator="S",
        n_examples=n,
        batch_size=8,
        max_updates=50,
        source_hashes={"loop": "x"},
    )


def _trainer(root, u, obs, nrm, contract):
    t = RecoverableTrainer(
        root,
        contract,
        u,
        obs,
        None,
        nrm.mean,
        nrm.scale,
        nrm.b_ref_sq,
        device=torch.device("cpu"),
    )
    t.snapshot_every = 1
    return t


def test_interrupt_resume_matches_uninterrupted(tmp_path):
    configure_determinism()
    u, obs, nrm = _data()
    c = _contract(20)
    ta = _trainer(tmp_path / "a", u, obs, nrm, c)
    ta.train(6, start_new=True)
    wa = {k: v.detach().clone() for k, v in ta.model.state_dict().items()}

    tb = _trainer(tmp_path / "b", u, obs, nrm, c)
    tb.train(3, start_new=True)
    tb2 = _trainer(tmp_path / "b", u, obs, nrm, c)
    tb2.train(6, start_new=False)
    assert tb2.updates == 6
    for k, v in wa.items():
        assert torch.allclose(v, tb2.model.state_dict()[k], atol=1e-5, rtol=1e-5), k


def test_bootstrap_keeps_year_multiplicity():
    from prigm.scoring import month_balanced_mean_replicated

    # one window per year-month 2005-08 and 2006-08 with scores 1 and 3
    values = np.array([1.0, 3.0])
    months = np.array([200508, 200608])

    def yof(m):
        return int(m) // 100

    # draw 2005 twice: month-balanced mean of {1} and {1} is 1, not unique-year mix
    m = month_balanced_mean_replicated(values, months, yof, np.array([2005, 2005]))
    assert m == 1.0
    mixed = month_balanced_mean_replicated(values, months, yof, np.array([2005, 2006]))
    assert mixed == 2.0


def test_test_barrier_blocks_without_receipt(tmp_path):
    from prigm.barrier import TestBarrierError, require_test_started

    require_test_started(tmp_path, "VAL")
    with pytest.raises(TestBarrierError):
        require_test_started(tmp_path, "TEST")
    # A bare receipt no longer authorizes test work without protocol/selection binding.
    (tmp_path / "TEST_STARTED.json").write_text('{"phase":"TEST_STARTED"}\n')
    with pytest.raises(TestBarrierError):
        require_test_started(tmp_path, "TEST")
    import json
    from prigm.recovery import sha256_file

    protocol = tmp_path / "PROTOCOL.json"
    protocol.write_text(json.dumps({"schema_version": 2, "exposure": "historical_test_exposed"}))
    selection = tmp_path / "SELECTION.json"
    selection.write_text('{"selected": "D"}\n')
    digest = sha256_file(protocol)
    (tmp_path / "TEST_STARTED.json").write_text(json.dumps({
        "phase": "TEST_STARTED", "protocol_sha256": digest,
        "selection_sha256": sha256_file(selection), "previous_test_exposure": True,
    }))
    assert require_test_started(tmp_path, "TEST", protocol_hash=digest) == {"selected": "D"}


def test_fresh_root_required(tmp_path):
    u, obs, nrm = _data(n=16, seed=0)
    c = _contract(16)
    t = _trainer(tmp_path / "c", u, obs, nrm, c)
    t.train(1, start_new=True)
    t2 = _trainer(tmp_path / "c", u, obs, nrm, c)
    with pytest.raises(PublicationError):
        t2.train(1, start_new=True)
