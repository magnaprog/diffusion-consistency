"""Restricted training-state serialization. No pickled NumPy objects."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch

from prigm.recovery import PublicationError


def tensor_state_to_cpu(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _restricted_types(obj) -> None:
    if torch.is_tensor(obj) or type(obj) in (str, int, float, bool, bytes, type(None)):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _restricted_types(k)
            _restricted_types(v)
    elif type(obj) in (list, tuple):
        for v in obj:
            _restricted_types(v)
    else:
        raise PublicationError(f"unsupported checkpoint type: {type(obj).__name__}")


def save_torch_payload(obj: dict) -> bytes:
    _restricted_types(obj)
    if not isinstance(obj, dict):
        raise PublicationError("checkpoint must be a dict")
    buf = io.BytesIO()
    torch.save(obj, buf, pickle_protocol=2)
    return buf.getvalue()


def load_torch_payload(data: bytes, map_location="cpu") -> dict:
    """Restricted loading also supports tensor/primitive optimizer and RNG state."""
    try:
        obj = torch.load(io.BytesIO(data), map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise PublicationError("unsupported or corrupt restricted checkpoint") from exc
    # torch's archive reader reports malformed ZIP files as RuntimeError.
    except RuntimeError as exc:
        if "PytorchStreamReader" not in str(exc) and "Invalid magic number" not in str(exc):
            raise
        raise PublicationError("corrupt checkpoint archive") from exc
    _restricted_types(obj)
    if not isinstance(obj, dict):
        raise PublicationError("checkpoint must be a dict")
    return obj


def numpy_rng_state(rng: np.random.Generator) -> dict:
    st = rng.bit_generator.state
    return json.loads(json.dumps(st, default=lambda x: x.tolist() if hasattr(x, "tolist") else x))


def restore_numpy_rng(state: dict) -> np.random.Generator:
    bg = np.random.PCG64()
    bg.state = state
    return np.random.Generator(bg)


def normalization_hash(mean, scale, b_ref_sq, n_bins) -> str:
    """SHA256 of compact sorted JSON with float64 values and an integer bin count.

    n_bins counts the normalizer's training-bin union, not the window length.
    """
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if mean.ndim != 1 or mean.shape != scale.shape:
        raise ValueError("normalization mean/scale shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("normalization must be finite with positive scales")
    b_ref_sq = float(np.float64(b_ref_sq))
    if not np.isfinite(b_ref_sq) or b_ref_sq <= 0 or int(n_bins) != n_bins or n_bins < 0:
        raise ValueError("invalid normalization reference or bin count")
    record = {"mean": mean.tolist(), "scale": scale.tolist(),
              "b_ref_sq": b_ref_sq, "n_bins": int(n_bins)}
    data = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash the shape and canonical little-endian float32 training values."""
    array = np.ascontiguousarray(array, dtype="<f4")
    h = hashlib.sha256(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def source_hashes(paths: list[Path], relative_to: Path | None = None) -> dict[str, str]:
    out = {}
    for p in paths:
        p = Path(p)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        out[p.relative_to(relative_to).as_posix() if relative_to is not None else str(p)] = h
    return out


def frozen_source_snapshot(root: Path) -> tuple[dict[str, str], bytes]:
    """Freeze every prigm Python source, addressed relative to the repository."""
    root = Path(root)
    paths = sorted((root / "prigm").rglob("*.py"))
    hashes = source_hashes(paths, relative_to=root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, expected in hashes.items():
            data = (root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise PublicationError(f"source changed while freezing: {name}")
            archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)
    return hashes, buf.getvalue()


def verify_source_snapshot(data: bytes, hashes: dict, expected_hash: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise PublicationError("source snapshot hash mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sorted(archive.namelist()) != sorted(hashes):
                raise PublicationError("source snapshot file list mismatch")
            for name, expected in hashes.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise PublicationError(f"source snapshot hash mismatch: {name}")
    except zipfile.BadZipFile as exc:
        raise PublicationError("invalid source snapshot archive") from exc


def weights_only_is_not_resume(payload: dict) -> None:
    """Acceptance: a model-only dict cannot satisfy the training-resume contract."""
    required = ("model", "optimizer", "updates", "epochs", "cursor", "rng")
    missing = [k for k in required if k not in payload]
    if missing:
        raise PublicationError(f"weights-only payload missing {missing}")
