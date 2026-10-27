"""Check CUDA inference with small checkpoints created for these tests.

The checkpoints contain initialized weights with zero optimizer updates.
Separately seeded toy data provide normalization and evaluation inputs.
When these tests run, observation hooks inspect the real CUDA U-Net and
sampling CLI, including masked hidden values and unchanged observed context.

Repeated and resumed outputs are compared only for the specified chunk
layout, including its final smaller batch. These fixtures do not train
scientific models or use production predictions, and their byte comparisons
do not establish equality across arbitrary GPU batch shapes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prigm.artifacts import (
    NOISE_CONVENTION, STATE_CONVENTION, json_bytes, load_arrays, read_ensemble,
)
from prigm.checkpoint import frozen_source_snapshot, save_torch_payload
from prigm.config import HIDDEN_END, HIDDEN_START, N_BINS, N_DRAWS, SAMPLE_CHUNK_WINDOWS
from prigm.model import ConditionalUNet
from prigm.recovery import (
    ContractMismatch, PublicationError, RunLock, publish_generation, sha256_bytes, sha256_file,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sampling integration requires CUDA")
READY = b"PRIGM_SAMPLING_READY"
CHILD_TIMEOUT = 120


def _sampling_fixture(root, channels, split):
    """A pinned v2, initialization-only checkpoint and independently seeded toy splits."""
    from prigm.normalize import fit_normalizer
    from prigm.toy_windows import make_toy_training

    train, _, _ = make_toy_training(20, seed=11)
    z, _, _ = make_toy_training(9, seed=13 if split == "VAL" else 42)
    nrm = fit_normalizer(train.reshape(-1, 6))
    identities = {}
    for name, values in (("TRAIN", train), (split, z)):
        path = root / f"{name.lower()}.npz"
        ids = [f"cuda-fixture-{name.lower()}-{i:02d}" for i in range(len(values))]
        np.savez(path, z=values, ids=np.asarray(ids), split=np.array(name),
                 mean=nrm.mean, scale=nrm.scale, b_ref_sq=np.array(nrm.b_ref_sq),
                 n_bins=np.array(nrm.n_bins))
        identities[name] = load_arrays(path, name)[3]
    source, snapshot = frozen_source_snapshot(ROOT)
    assert "prigm/run_eval.py" in source
    contract = {
        "contract_version": 2, "arm": "G5" if channels == 5 else "D", "seed": 7,
        "n_generated": channels, "n_aux": 0, "widths": [8, 16, 16],
        "c_r": 0.0, "residual": None, "estimator": "S",
        "n_examples": len(train), "batch_size": 8, "max_updates": 7,
        "source_hashes": source, "data_hashes": {"arrays_sha256": identities["TRAIN"]["sha256"]},
        "normalization_hash": identities[split]["normalization_sha256"],
        "settings": {"source_snapshot_sha256": sha256_bytes(snapshot)},
    }
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        model = ConditionalUNet(n_generated=channels, widths=(8, 16, 16))
    counters = {"updates": 0, "epochs": 0, "cursor": 0}
    payload = save_torch_payload({
        "model": model.state_dict(), "contract": contract, **counters,
        "normalization": {
            "mean": torch.from_numpy(nrm.mean.copy()), "scale": torch.from_numpy(nrm.scale.copy()),
            "b_ref_sq": nrm.b_ref_sq, "n_bins": nrm.n_bins,
        },
    })
    run = root / "run"
    run.mkdir()
    (run / "source_snapshot.zip").write_bytes(snapshot)
    with RunLock(run):
        publish_generation(run, 0, {"state.pt": payload}, {"contract": contract, **counters})
    checkpoint_hash = sha256_file(run / "gen_00000000/state.pt")
    protocol = {
        "schema_version": 2, "exposure": "historical_test_exposed",
        "training_source_hashes": source, "inference_source_hashes": source,
        "training_data_hashes": contract["data_hashes"], "arrays": identities,
        "samplers": {
            "VAL": {"n_eval": 25, "n_draws": 39, "eval_seed": 7},
            "TEST": {"n_eval": 50, "n_draws": 39, "eval_seed": 7},
        },
    }
    (root / "PROTOCOL.json").write_bytes(json_bytes(protocol))
    protocol_hash = sha256_file(root / "PROTOCOL.json")
    if split == "TEST":
        (root / "SELECTION.json").write_bytes(json_bytes({"test_checkpoint_hashes": [checkpoint_hash]}))
        (root / "TEST_STARTED.json").write_bytes(json_bytes({
            "phase": "TEST_STARTED", "protocol_sha256": protocol_hash,
            "selection_sha256": sha256_file(root / "SELECTION.json"), "previous_test_exposure": True,
        }))
    return {
        "run-root": str(run), "generation": 0, "checkpoint-sha256": checkpoint_hash,
        "arrays": str(root / f"{split.lower()}.npz"), "split": split,
        "n-eval": protocol["samplers"][split]["n_eval"], "eval-seed": 7,
        "study-root": str(root), "protocol-sha256": protocol_hash, "limit": 8,
    }


def _cli_args(options):
    return [item for key, value in options.items() for item in (f"--{key}", str(value))]


def _eval_child(options, interrupt):
    """Run the real CLI with observation-only hooks on the real CUDA U-Net."""
    import prigm.recovery as recovery
    from prigm.run_eval import main
    from prigm.schedule import ddim_timesteps

    assert os.environ["PRIGM_REQUIRE_CUDA"] == "1"
    assert torch.cuda.is_available(), "child must not silently sample on CPU"
    torch.set_num_threads(1)
    z, _, nrm, _ = load_arrays(Path(options["arrays"]), options["split"])
    offset = options["offset"]
    z = z[offset:offset + options["limit"]]
    # Independent of physical_to_model: six conditioning channels, hidden truth zeroed.
    clean = ((z - nrm.mean) / nrm.scale).transpose(0, 2, 1).astype(np.float32)
    clean[:, :, HIDDEN_START:HIDDEN_END] = 0.0
    expected = None  # Allocate only after run_eval configures CUDA determinism.
    steps = ddim_timesteps(options["n-eval"]).tolist()
    calls = []

    def inspect_forward(model, inputs):
        nonlocal expected
        if not isinstance(model, ConditionalUNet):
            return
        noisy, observed, mask, relpos, timestep = inputs
        assert not model.training and not torch.is_grad_enabled()
        assert all(x.device.type == "cuda" for x in inputs)
        assert next(model.parameters()).device.type == "cuda"
        assert noisy.shape == (len(z) * 39, model.n_generated, 48)
        assert noisy.dtype == observed.dtype == torch.float32
        if expected is None:
            expected = torch.as_tensor(clean, device=observed.device).repeat_interleave(39, dim=0)
        assert torch.equal(observed, expected), "conditioning changed or leaked hidden truth"
        for flank in (slice(None, HIDDEN_START), slice(HIDDEN_END, None)):
            assert torch.equal(noisy[:, :, flank], expected[:, :model.n_generated, flank]), "noised observed flank"
        expected_mask = torch.ones_like(mask)
        expected_mask[:, :, HIDDEN_START:HIDDEN_END] = 0.0
        assert torch.equal(mask, expected_mask)
        assert torch.isfinite(noisy).all().item() and torch.isfinite(relpos).all().item()
        assert len(calls) < len(steps)
        assert torch.equal(timestep, torch.full_like(timestep, steps[len(calls)]))
        calls.append(steps[len(calls)])

    commit_path = Path(options["out"]) / ".tmp_gen_00000000" / "COMMIT.json"
    original_write = recovery.atomic_write_bytes

    def publication_hook(path, data):
        if Path(path) == commit_path:
            manifest = json.loads(data)
            assert manifest["generation"] == 0 and set(manifest["files"]) == {"ensemble.npz"}
            assert (commit_path.parent / "ensemble.npz").is_file()
            assert not (commit_path.parent / "state.pt").exists()
            assert calls == steps, "CLI did not execute the full 25/50-step sampler"
            if interrupt:
                print(READY.decode(), flush=True)
                sys.stdin.buffer.read(1)
                raise AssertionError("parent must SIGKILL the owned child before publication")
        return original_write(path, data)

    hook = torch.nn.modules.module.register_module_forward_pre_hook(inspect_forward)
    recovery.atomic_write_bytes = publication_hook
    try:
        status = main(_cli_args(options))
        assert status == 0
        assert calls == steps
    finally:
        hook.remove()
        recovery.atomic_write_bytes = original_write


def _run_child(options, interrupt=False):
    from experiments.run_prigm_repair import preserve_uncommitted

    code = (
        "import json, sys; from tests.test_prigm_cuda_sampling import _eval_child; "
        "_eval_child(json.loads(sys.argv[1]), sys.argv[2] == 'kill')"
    )
    command = [sys.executable, "-B", "-u", "-c", code, json.dumps(options), "kill" if interrupt else "run"]
    env = os.environ.copy()  # Inherit CUDA_VISIBLE_DEVICES; never choose a physical GPU here.
    env["PRIGM_REQUIRE_CUDA"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = bytearray()
    deadline = time.monotonic() + CHILD_TIMEOUT

    def failure(message):
        pytest.fail(f"{message}; out={options['out']}\nchild log:\n{log.decode(errors='replace')}")

    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(timeout=remaining):
                    failure(f"CUDA child exceeded {CHILD_TIMEOUT}s waiting for output/READY")
                data = os.read(process.stdout.fileno(), 65536)
                if not data:
                    break
                log.extend(data)
                if interrupt and READY in log.splitlines():
                    output = Path(options["out"])
                    with pytest.raises(PublicationError, match="missing or invalid JSON"):
                        read_ensemble(output)
                    # Publication still owns WRITE.lock while stopped at the handshake.
                    with pytest.raises(PublicationError, match="locked"):
                        preserve_uncommitted(output)
                    process.kill()  # Only this Popen child, never a process group or GPU job.
                    rest, _ = process.communicate(timeout=10)
                    log.extend(rest)
                    if process.returncode != -signal.SIGKILL:
                        failure(f"expected SIGKILL, got {process.returncode}")
                    return
            status = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            if interrupt or status != 0:
                failure(f"child exited {status}" + (" without READY" if interrupt else ""))
    except subprocess.TimeoutExpired:
        failure("CUDA child did not exit within the bounded wait")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()


@pytest.mark.parametrize("channels", [5, 6], ids=["C5", "C6"])
@pytest.mark.parametrize("split,n_eval", [("VAL", 25), ("TEST", 50)], ids=["VAL25", "TEST50"])
def test_cuda_cli_sampling_redraw_after_sigkill(tmp_path, channels, split, n_eval):
    from experiments.run_prigm_repair import preserve_uncommitted
    from prigm.run_eval import main

    assert (SAMPLE_CHUNK_WINDOWS, N_DRAWS, N_BINS) == (8, 39, 48)
    options = _sampling_fixture(tmp_path, channels, split)
    assert options["n-eval"] == n_eval
    _, ids, _, identity = load_arrays(Path(options["arrays"]), split)
    assert len(ids) == len(set(ids)) == 9
    oversized = tmp_path / "oversized"
    with pytest.raises(ContractMismatch, match="invalid fixed-layout chunk bounds"):
        main(_cli_args({**options, "out": str(oversized), "offset": 0, "limit": 9}))
    assert not oversized.exists()

    expected_parts, actual_parts, actual_ids = [], [], []
    for offset, count in ((0, 8), (8, 1)):
        baseline = tmp_path / f"baseline-{offset}"
        redraw = tmp_path / f"redraw-{offset}"
        _run_child({**options, "out": str(baseline), "offset": offset})
        if offset == 0:
            _run_child({**options, "out": str(redraw), "offset": offset}, interrupt=True)
            pending = redraw / ".tmp_gen_00000000" / "ensemble.npz"
            interrupted_bytes = pending.read_bytes()
            assert not (pending.parent / "COMMIT.json").exists()
            assert not (redraw / "gen_00000000").exists()
            assert not (redraw / "LATEST").exists()
            with pytest.raises(PublicationError, match="missing or invalid JSON"):
                read_ensemble(redraw)
            # The coordinator owns its lock; preserve_uncommitted acquires the chunk lock.
            with RunLock(tmp_path):
                preserve_uncommitted(redraw)
            preserved = redraw.with_name(f"{redraw.name}.incomplete-1")
            assert not redraw.exists()
            preserved_payload = preserved / ".tmp_gen_00000000" / "ensemble.npz"
            assert preserved_payload.read_bytes() == interrupted_bytes
            assert json.loads((preserved / "PRESERVED.json").read_text())["original"] == redraw.name
            with pytest.raises(PublicationError, match="missing or invalid JSON"):
                read_ensemble(preserved)

        _run_child({**options, "out": str(redraw), "offset": offset})
        expected, expected_ids, expected_meta = read_ensemble(baseline)
        actual, sampled_ids, meta = read_ensemble(redraw)
        assert sampled_ids == expected_ids == ids[offset:offset + count]
        assert actual.shape == (count, 39, channels, 48) and actual.dtype == np.float32
        assert np.isfinite(actual).all()
        assert meta == expected_meta
        assert meta["device"].startswith("cuda")
        assert meta["arrays"] == identity and meta["checkpoint_sha256"] == options["checkpoint-sha256"]
        assert meta["protocol_sha256"] == options["protocol-sha256"]
        assert (meta["offset"], meta["count"], meta["n_generated"]) == (offset, count, channels)
        assert meta["sampler"] == {"n_eval": n_eval, "n_draws": 39, "eval_seed": 7}
        assert meta["state_convention"] == STATE_CONVENTION and meta["noise_convention"] == NOISE_CONVENTION
        # Stored ensembles are hidden-only; clean flanks were checked inside every forward.
        assert not np.any(actual[..., :HIDDEN_START]) and not np.any(actual[..., HIDDEN_END:])
        assert np.any(actual[:, 1:, :, HIDDEN_START:HIDDEN_END] != actual[:, :1, :, HIDDEN_START:HIDDEN_END])
        np.testing.assert_array_equal(actual, expected)
        baseline_bytes = (baseline / "gen_00000000/ensemble.npz").read_bytes()
        assert (redraw / "gen_00000000/ensemble.npz").read_bytes() == baseline_bytes
        if offset == 0:
            assert interrupted_bytes == baseline_bytes == preserved_payload.read_bytes()
        expected_parts.append(expected)
        actual_parts.append(actual)
        actual_ids.extend(sampled_ids)

    # Compare all nine windows only with the identical 8+1 layout, not another GPU batch shape.
    assert actual_ids == ids
    expected, actual = np.concatenate(expected_parts), np.concatenate(actual_parts)
    np.testing.assert_array_equal(actual, expected)
    assert actual.tobytes() == expected.tobytes()
