"""Recoverable chronological training (plan Section 10)."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from prigm.checkpoint import (
    array_hash,
    load_torch_payload,
    normalization_hash,
    numpy_rng_state,
    restore_numpy_rng,
    save_torch_payload,
    tensor_state_to_cpu,
    verify_source_snapshot,
    weights_only_is_not_resume,
)
from prigm.config import (
    HIDDEN_END,
    HIDDEN_START,
    N_BINS,
    N_CHANNELS,
    N_DIFFUSION_STEPS,
    RECOVERY_EVERY,
)
from prigm.loss import LossConfig, prigm_loss
from prigm.model import ConditionalUNet
from prigm.recovery import (
    ContractMismatch,
    PublicationError,
    RunLock,
    atomic_write_bytes,
    fsync_dir,
    load_committed,
    publish_generation,
    read_json,
)
from prigm.schedule import build_schedule
from prigm.train import clip_and_step, make_optimizer, require_cuda_if_requested


def _capture_rng(noise_gen: torch.Generator, np_rng: np.random.Generator, device: torch.device) -> dict:
    py = random.getstate()
    out = {
        "python": [py[0], list(py[1]), py[2]],
        "numpy": numpy_rng_state(np_rng),
        "torch": torch.get_rng_state().cpu().numpy().tobytes().hex(),
        "noise_gen": noise_gen.get_state().cpu().numpy().tobytes().hex(),
    }
    if device.type == "cuda":
        out["cuda"] = [t.cpu().numpy().tobytes().hex() for t in torch.cuda.get_rng_state_all()]
    return out


def _hex_to_tensor(h: str) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(bytes.fromhex(h), dtype=np.uint8).copy())


def _restore_rng(blob: dict, noise_gen: torch.Generator, device: torch.device) -> np.random.Generator:
    py = blob["python"]
    gauss = None if py[2] is None else float(py[2])
    random.setstate((int(py[0]), tuple(int(x) for x in py[1]), gauss))
    np_rng = restore_numpy_rng(blob["numpy"])
    torch.set_rng_state(_hex_to_tensor(blob["torch"]))
    noise_gen.set_state(_hex_to_tensor(blob["noise_gen"]))
    if device.type == "cuda" and "cuda" in blob:
        torch.cuda.set_rng_state_all([_hex_to_tensor(x) for x in blob["cuda"]])
    return np_rng


def _require_finite(obj) -> None:
    if torch.is_tensor(obj):
        if not torch.isfinite(obj).all().item():
            raise PublicationError("nonfinite checkpoint tensor")
    elif isinstance(obj, dict):
        for value in obj.values():
            _require_finite(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _require_finite(value)
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise PublicationError("nonfinite checkpoint scalar")


@dataclass
class RunContract:
    arm: str
    seed: int
    n_generated: int
    n_aux: int
    widths: tuple[int, ...]
    c_r: float
    residual: str | None
    estimator: str
    n_examples: int
    batch_size: int
    max_updates: int
    source_hashes: dict
    contract_version: int = 2
    data_hashes: dict = field(default_factory=dict)
    normalization_hash: str = ""
    settings: dict = field(default_factory=dict)


def contract_dict(c: RunContract) -> dict:
    d = asdict(c)
    d["widths"] = list(c.widths)
    return d


class RecoverableTrainer:
    def __init__(
        self,
        run_root: Path,
        contract: RunContract,
        u0: np.ndarray,
        observed: np.ndarray,
        aux: np.ndarray | None,
        mean: np.ndarray,
        scale: np.ndarray,
        b_ref_sq: float,
        device: torch.device | None = None,
        *,
        normalization_n_bins: int = 0,
        source_snapshot: bytes | None = None,
    ):
        self.run_root = Path(run_root)
        self.contract = contract
        if contract.contract_version != 2:
            raise ContractMismatch("only contract version 2 is supported")
        if contract.n_aux != 0 or aux is not None:
            raise ContractMismatch("auxiliary conditioning is not implemented")
        if contract.batch_size <= 0 or contract.max_updates < 0:
            raise ContractMismatch("invalid batch size or training budget")
        self.device = device or require_cuda_if_requested()
        self.u0_np = np.asarray(u0, dtype=np.float32)  # (N, C, L)
        self.obs_np = np.asarray(observed, dtype=np.float32)
        self.n = int(self.u0_np.shape[0])
        if self.n <= 0 or self.n != contract.n_examples:
            raise ContractMismatch("n_examples mismatch or empty data")
        if (self.u0_np.shape != (self.n, contract.n_generated, N_BINS)
                or self.obs_np.shape != (self.n, N_CHANNELS, N_BINS)):
            raise ContractMismatch("training array shape mismatch")
        if not np.isfinite(self.u0_np).all() or not np.isfinite(self.obs_np).all():
            raise ContractMismatch("nonfinite training data")
        self.data_hashes = {"u0_sha256": array_hash(self.u0_np),
                            "observed_sha256": array_hash(self.obs_np)}
        self.normalization = {
            "mean": torch.as_tensor(np.asarray(mean, dtype=np.float64).copy()),
            "scale": torch.as_tensor(np.asarray(scale, dtype=np.float64).copy()),
            "b_ref_sq": float(b_ref_sq), "n_bins": int(normalization_n_bins),
        }
        if self.normalization["mean"].shape != (N_CHANNELS,):
            raise ContractMismatch("normalization requires six channels")
        self.normalization_digest = normalization_hash(mean, scale, b_ref_sq, normalization_n_bins)
        if contract.normalization_hash and contract.normalization_hash != self.normalization_digest:
            raise ContractMismatch("normalization hash mismatch")
        self.source_snapshot = source_snapshot
        self.snapshot_every = contract.settings.get("recovery_every", RECOVERY_EVERY)
        self.milestones = set(contract.settings.get("milestones", []))
        if any(type(v) is not int or not 0 < v <= contract.max_updates for v in self.milestones):
            raise ContractMismatch("invalid milestone updates")
        torch.manual_seed(contract.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(contract.seed)
        self.noise_gen = torch.Generator(device="cpu")
        self.noise_gen.manual_seed(contract.seed + 17)
        self.np_rng = np.random.Generator(np.random.PCG64(contract.seed + 31))
        self.model = ConditionalUNet(
            n_generated=contract.n_generated,
            widths=tuple(contract.widths),
        ).to(self.device)
        self.opt = make_optimizer(self.model)
        self.sched = build_schedule()
        self.cfg = LossConfig(
            c_r=contract.c_r,
            residual=contract.residual,
            estimator=contract.estimator,
            n_generated=contract.n_generated,
            b_ref_sq=b_ref_sq,
            mean=np.asarray(mean, dtype=np.float64).copy(),
            scale=np.asarray(scale, dtype=np.float64).copy(),
        )
        self.mask = torch.ones(1, 1, N_BINS, device=self.device)
        self.mask[:, :, HIDDEN_START:HIDDEN_END] = 0.0
        t = np.arange(N_BINS, dtype=np.float32)
        center = 0.5 * (N_BINS - 1)
        self.rel = torch.as_tensor(((t - center) / center)[None, None, :], device=self.device)
        self.updates = 0
        self.epochs = 0
        self.cursor = 0
        self.stop_flag = False
        self._run_lock = None
        self._published_update = None

    def _check_training_data(self):
        for name, digest in self.data_hashes.items():
            if name in self.contract.data_hashes and self.contract.data_hashes[name] != digest:
                raise ContractMismatch(f"training data hash mismatch: {name}")

    def _batch_tensors(self, sl: slice):
        u0 = torch.as_tensor(self.u0_np[sl], device=self.device)
        obs = torch.as_tensor(self.obs_np[sl], device=self.device)
        b = u0.shape[0]
        return u0, obs, self.mask.expand(b, -1, -1), self.rel.expand(b, -1, -1)

    def _state_blob(self) -> dict:
        return {
            "model": tensor_state_to_cpu(self.model.state_dict()),
            "optimizer": self.opt.state_dict(),
            "updates": self.updates,
            "epochs": self.epochs,
            "cursor": self.cursor,
            "rng": _capture_rng(self.noise_gen, self.np_rng, self.device),
            "contract": contract_dict(self.contract),
            "normalization": self.normalization,
            "data_hashes": self.data_hashes,
        }

    def snapshot_payload(self) -> dict[str, bytes]:
        blob = self._state_blob()
        self._validate_blob(blob, self._metadata("snapshot"), inference=False)
        return {
            "state.pt": save_torch_payload(blob),
            "meta.json": (json.dumps(contract_dict(self.contract), sort_keys=True) + "\n").encode(),
        }

    def _metadata(self, kind):
        return {"kind": kind, "generation": self.updates, "updates": self.updates,
                "epochs": self.epochs, "cursor": self.cursor, "contract": contract_dict(self.contract)}

    def publish(self, generation: int, kind: str) -> None:
        if self._run_lock is None or self._run_lock.fd is None:
            raise PublicationError("publication requires the training run lock")
        if generation != self.updates:
            raise PublicationError("generation must equal completed updates")
        if self._published_update == self.updates:
            return
        publish_generation(self.run_root, generation, self.snapshot_payload(), self._metadata(kind))
        self._published_update = self.updates

    def _validate_blob(self, blob: dict, meta: dict, *, inference: bool) -> None:
        weights_only_is_not_resume(blob)
        try:
            saved = blob["contract"]
            if saved != meta["contract"]:
                raise PublicationError("payload contract differs from commit metadata")
            n, bs = saved["n_examples"], saved["batch_size"]
            updates, epochs, cursor = (blob[k] for k in ("updates", "epochs", "cursor"))
            if (any(type(v) is not int for v in (n, bs, updates, epochs, cursor, saved["max_updates"]))
                    or n <= 0 or bs <= 0 or not 0 <= updates <= saved["max_updates"]):
                raise PublicationError("invalid checkpoint counters")
            steps_per_epoch = (n + bs - 1) // bs
            if epochs != updates // steps_per_epoch or cursor != (updates % steps_per_epoch) * bs:
                raise PublicationError("checkpoint step/epoch/cursor inconsistency")
            if (meta["generation"] != updates or any(type(meta[k]) is not int or meta[k] != blob[k]
                                                    for k in ("updates", "epochs", "cursor"))):
                raise PublicationError("checkpoint counters differ from commit metadata")
            nrm = blob["normalization"]
            if any(not torch.is_tensor(nrm[k]) or nrm[k].dtype != torch.float64
                   or nrm[k].shape != (N_CHANNELS,) for k in ("mean", "scale")):
                raise PublicationError("checkpoint normalization requires six float64 channels")
            digest = normalization_hash(nrm["mean"], nrm["scale"], nrm["b_ref_sq"], nrm["n_bins"])
            if saved.get("normalization_hash") and digest != saved["normalization_hash"]:
                raise PublicationError("checkpoint normalization hash mismatch")
            for name, value in blob["data_hashes"].items():
                if name in saved.get("data_hashes", {}) and saved["data_hashes"][name] != value:
                    raise PublicationError("checkpoint data hashes differ from contract")
            _require_finite(blob)
            cur = contract_dict(self.contract)
            excluded = {"n_examples"} if inference else set()
            if ({k: v for k, v in saved.items() if k not in excluded}
                    != {k: v for k, v in cur.items() if k not in excluded}):
                raise ContractMismatch("scientific contract changed")
            if digest != self.normalization_digest:
                raise ContractMismatch("normalization changed")
            if not inference and blob["data_hashes"] != self.data_hashes:
                raise ContractMismatch("training data changed")
            model_state = self.model.state_dict()
            if blob["model"].keys() != model_state.keys():
                raise PublicationError("model state keys mismatch")
            for name, tensor in blob["model"].items():
                if tensor.shape != model_state[name].shape or tensor.dtype != model_state[name].dtype:
                    raise PublicationError(f"model tensor mismatch: {name}")
            optimizer = blob["optimizer"]
            if optimizer["param_groups"] != self.opt.state_dict()["param_groups"]:
                raise PublicationError("optimizer parameter groups mismatch")
            params = list(self.model.parameters())
            states = optimizer["state"]
            expected_ids = set(range(len(params))) if updates else set()
            if set(states) != expected_ids:
                raise PublicationError("optimizer state is incomplete")
            for i, state in states.items():
                if set(state) != {"step", "exp_avg", "exp_avg_sq"} or state["step"].item() != updates:
                    raise PublicationError("optimizer step mismatch")
                for key in ("exp_avg", "exp_avg_sq"):
                    if state[key].shape != params[i].shape or state[key].dtype != params[i].dtype:
                        raise PublicationError("optimizer moment shape/dtype mismatch")
            rng = blob["rng"]
            py = rng["python"]
            random.Random().setstate((py[0], tuple(py[1]), py[2]))
            restore_numpy_rng(rng["numpy"])
            for key in ("torch", "noise_gen"):
                state = _hex_to_tensor(rng[key])
                if state.shape != torch.get_rng_state().shape:
                    raise PublicationError("invalid CPU RNG state length")
                try:
                    torch.Generator(device="cpu").set_state(state)
                except RuntimeError as exc:
                    raise PublicationError("invalid CPU RNG state") from exc
            if not inference and self.device.type == "cuda":
                if len(rng["cuda"]) != torch.cuda.device_count():
                    raise ContractMismatch("CUDA RNG device count changed")
                for i, encoded in enumerate(rng["cuda"]):
                    if _hex_to_tensor(encoded).shape != torch.cuda.get_rng_state(i).shape:
                        raise PublicationError("invalid CUDA RNG state length")
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            raise PublicationError("malformed checkpoint state") from exc

    def _verify_sources(self) -> None:
        expected = self.contract.settings.get("source_snapshot_sha256")
        if expected:
            try:
                data = (self.run_root / "source_snapshot.zip").read_bytes()
            except FileNotFoundError as exc:
                raise PublicationError("missing frozen source snapshot") from exc
            verify_source_snapshot(data, self.contract.source_hashes, expected)

    def load(self, generation: int | None = None, *, inference: bool = False) -> dict:
        if not inference:
            self._check_training_data()
        blob = None

        def validate(meta):
            nonlocal blob
            gen_dir = self.run_root / f"gen_{meta['generation']:08d}"
            if not {"state.pt", "meta.json"}.issubset(meta["files"]):
                raise PublicationError("missing training payload entries")
            if read_json(gen_dir / "meta.json") != meta.get("contract"):
                raise PublicationError("meta.json differs from commit contract")
            blob = load_torch_payload((gen_dir / "state.pt").read_bytes(), map_location="cpu")
            self._validate_blob(blob, meta, inference=inference)

        meta = load_committed(self.run_root, generation, validator=validate)
        self._verify_sources()
        self.model.load_state_dict(blob["model"])
        self.model.to(self.device)
        self.updates = blob["updates"]
        self.epochs = blob["epochs"]
        self.cursor = 0 if inference else blob["cursor"]
        if not inference:
            self.opt.load_state_dict(blob["optimizer"])
            self.np_rng = _restore_rng(blob["rng"], self.noise_gen, self.device)
        self._published_update = self.updates
        return meta

    def _record_recovery(self, meta: dict, target: int) -> None:
        incomplete = sorted(p for p in self.run_root.glob(".tmp_gen_*")
                            if p.is_dir() and p.name[len(".tmp_gen_"):].isdigit())
        if "recovery" not in meta and not incomplete:
            return
        record = dict(meta.get("recovery", {"selected_generation": self.updates,
                      "replay_from_update": self.updates, "skipped": []}), total_target=target)
        stamp = time.time_ns()
        originals = [f"gen_{item['generation']:08d}" for item in record["skipped"]]
        originals.extend(p.name for p in incomplete)
        record["preserved"] = [{"original": name, "path": f"rejected_{name.lstrip('.')}_{stamp}"}
                               for name in originals]
        # Record the replay and preservation destinations before changing names.
        atomic_write_bytes(self.run_root / f"recovery_{stamp}.json",
                           (json.dumps(record, sort_keys=True, indent=2) + "\n").encode())
        for item in record["preserved"]:
            os.rename(self.run_root / item["original"], self.run_root / item["path"])
            fsync_dir(self.run_root)
        atomic_write_bytes(self.run_root / "LATEST", f"{self.updates}\n".encode())

    @contextmanager
    def _signal_stop(self):
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        old = {}

        def stop(signum, frame):
            self.stop_flag = True

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old[sig] = signal.signal(sig, stop)
            yield
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)

    def _one_update(self) -> dict:
        bs = self.contract.batch_size
        sl = slice(self.cursor, min(self.cursor + bs, self.n))
        n_ex = sl.stop - sl.start
        u0, obs, mask, rel = self._batch_tensors(sl)
        t = torch.randint(
            1, N_DIFFUSION_STEPS + 1, (n_ex,), generator=self.noise_gen, device="cpu"
        ).to(self.device)
        noise = torch.randn(u0.shape, generator=self.noise_gen, device="cpu").to(self.device)
        self.opt.zero_grad(set_to_none=True)
        a = torch.as_tensor(self.sched["a"], device=self.device, dtype=u0.dtype)[t][:, None, None]
        s = torch.as_tensor(self.sched["s"], device=self.device, dtype=u0.dtype)[t][:, None, None]
        hid = slice(HIDDEN_START, HIDDEN_END)
        u_t = u0.clone()
        u_t[:, :, hid] = a * u0[:, :, hid] + s * noise[:, :, hid]
        out = prigm_loss(self.model, u0, u_t, t, obs, mask, rel, self.cfg, self.sched)
        if not torch.isfinite(out["batch"]).item():
            raise PublicationError("nonfinite training loss")
        out["batch"].backward()
        gn = clip_and_step(self.model, self.opt)
        if not math.isfinite(gn) or not torch.stack([
            torch.isfinite(p).all() for p in self.model.parameters()
        ]).all().item():
            raise PublicationError("nonfinite gradient or model parameters")
        self.cursor = sl.stop
        if self.cursor >= self.n:
            self.cursor = 0
            self.epochs += 1
        self.updates += 1
        out["grad_norm"] = gn
        return out

    def train(self, total_target: int, start_new: bool) -> None:
        """Train until total_target updates have been completed since the run began.

        On resume, total_target includes updates already present in the checkpoint;
        it is not a number of additional updates. The contract's max_updates is the
        run's fixed lifetime limit. One writer lock covers loading, training, and
        publication of checkpoints. Checkpoints retain optimizer and random-number
        state as well as model parameters so interrupted work can resume.
        """
        if type(total_target) is not int or not 0 <= total_target <= self.contract.max_updates:
            raise ValueError("total target must be within the frozen training budget")
        if self.snapshot_every <= 0:
            raise ValueError("snapshot interval must be positive")
        self._check_training_data()
        if start_new and self.run_root.exists() and any(self.run_root.iterdir()):
            raise PublicationError("run root not fresh")
        with RunLock(self.run_root) as lock, self._signal_stop():
            self._run_lock = lock
            try:
                if start_new:
                    if any(p.name != "WRITE.lock" for p in self.run_root.iterdir()):
                        raise PublicationError("run root not fresh")
                    if self.updates != 0:
                        raise PublicationError("new training requires initial state")
                    expected = self.contract.settings.get("source_snapshot_sha256")
                    if expected:
                        if self.source_snapshot is None:
                            raise PublicationError("new training requires a frozen source snapshot")
                        verify_source_snapshot(self.source_snapshot, self.contract.source_hashes, expected)
                        atomic_write_bytes(self.run_root / "source_snapshot.zip", self.source_snapshot)
                    self.publish(0, "step-zero")
                else:
                    meta = self.load()
                    if total_target < self.updates:
                        raise ValueError("total target precedes the resumed update")
                    self._record_recovery(meta, total_target)
                while self.updates < total_target and not self.stop_flag:
                    self._one_update()
                    if self.updates in self.milestones:
                        self.publish(self.updates, "milestone")
                    elif self.cursor == 0:
                        self.publish(self.updates, "epoch")
                    elif self.updates % self.snapshot_every == 0:
                        self.publish(self.updates, "interval")
                self.publish(self.updates, "graceful" if self.stop_flag else "target")
            finally:
                self._run_lock = None


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
