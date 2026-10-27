"""Identity and durable publication of PriGM evaluation artifacts."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from prigm.checkpoint import normalization_hash
from prigm.config import N_BINS, N_CHANNELS, N_DRAWS
from prigm.normalize import AffineNormalizer
from prigm.recovery import (
    ContractMismatch, PublicationError, RunLock, load_committed,
    publish_generation, sha256_bytes, sha256_file,
)

SCHEMA_VERSION = 2
STATE_CONVENTION = "clean_observed_flanks_hidden_diffusion"
NOISE_CONVENTION = "pcg64_canonical_six_channels_v2"


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def id_hash(ids: list[str]) -> str:
    return sha256_bytes(json.dumps(ids, ensure_ascii=True, separators=(",", ":")).encode())


def require_source_match(protocol: dict) -> None:
    source = {f"prigm/{p.name}": sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    if source != protocol["inference_source_hashes"]:
        raise ContractMismatch("analysis or inference source changed after protocol publication")


def load_arrays(path: Path, split: str, expected_sha256: str | None = None):
    path = Path(path)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ContractMismatch("array hash changed")
    with np.load(path, allow_pickle=False) as blob:
        required = {"z", "ids", "split", "mean", "scale", "b_ref_sq", "n_bins"}
        if not required.issubset(blob.files):
            raise ContractMismatch("arrays lack versioned split, IDs or normalization")
        z = blob["z"]
        ids_array = blob["ids"]
        actual_split = str(blob["split"].item()).upper()
        mean, scale = blob["mean"], blob["scale"]
        b_ref_sq = float(blob["b_ref_sq"].item())
        n_bins_value = blob["n_bins"].item()
        if not np.isfinite(n_bins_value) or n_bins_value != int(n_bins_value):
            raise ContractMismatch("normalization count must be finite and integral")
        n_bins = int(n_bins_value)
    if actual_split != split.upper():
        raise ContractMismatch("array split does not match requested split")
    if z.ndim != 3 or z.shape[1:] != (N_BINS, N_CHANNELS) or not len(z):
        raise ContractMismatch("invalid physical window shape")
    if ids_array.shape != (len(z),) or ids_array.dtype.kind != "U":
        raise ContractMismatch("window IDs must be one string per window")
    ids = ids_array.tolist()
    if len(set(ids)) != len(ids) or any(not wid for wid in ids):
        raise ContractMismatch("empty or duplicate array window IDs")
    if mean.shape != (N_CHANNELS,) or scale.shape != (N_CHANNELS,):
        raise ContractMismatch("invalid normalization shape")
    if (not np.isfinite(z).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(b_ref_sq) or b_ref_sq <= 0 or n_bins <= 0):
        raise ContractMismatch("nonfinite arrays or invalid training normalization")
    nrm = AffineNormalizer(mean=mean, scale=scale, b_ref_sq=b_ref_sq, n_bins=n_bins)
    identity = {
        "sha256": digest, "split": actual_split, "n_windows": len(ids),
        "ids_sha256": id_hash(ids),
        "normalization_sha256": normalization_hash(mean, scale, b_ref_sq, n_bins),
    }
    return z, ids, nrm, identity


def validate_ensemble(ensemble: np.ndarray, ids: list[str], meta: dict) -> None:
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ContractMismatch("unsupported ensemble schema")
    channels = meta.get("n_generated")
    if channels not in (5, 6):
        raise ContractMismatch("invalid generated channel count")
    expected = (len(ids), N_DRAWS, channels, N_BINS)
    procedure = meta.get("procedure", "diffusion")
    if procedure not in {"diffusion", "interpolation", "persistence", "gaussian"}:
        raise ContractMismatch("unknown predictive procedure")
    dtype = np.dtype("float32" if procedure == "diffusion" else "float64")
    if ensemble.shape != expected or ensemble.dtype != dtype:
        raise ContractMismatch(f"ensemble shape/dtype mismatch: expected {expected}, {dtype}")
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
        raise ContractMismatch("empty or duplicate ensemble IDs")
    if id_hash(ids) != meta.get("ids_sha256"):
        raise ContractMismatch("ensemble ID hash mismatch")
    if not np.isfinite(ensemble).all():
        raise PublicationError("nonfinite ensemble cannot be committed")
    if procedure == "diffusion":
        if meta.get("state_convention") != STATE_CONVENTION or meta.get("noise_convention") != NOISE_CONVENTION:
            raise ContractMismatch("ensemble sampling convention mismatch")
    elif meta.get("state_convention") != "physical_control_standardized_float64":
        raise ContractMismatch("control representation mismatch")


def publish_ensemble(root: Path, ensemble: np.ndarray, ids: list[str], meta: dict) -> Path:
    root = Path(root)
    if root.exists():
        raise PublicationError(f"output root exists: {root}")
    meta = {**meta, "schema_version": SCHEMA_VERSION, "ids_sha256": id_hash(ids)}
    validate_ensemble(ensemble, ids, meta)
    buffer = io.BytesIO()
    np.savez(buffer, ensemble=ensemble, ids=np.asarray(ids))
    root.mkdir(parents=True)
    with RunLock(root):
        return publish_generation(root, 0, {"ensemble.npz": buffer.getvalue()}, meta)


def read_ensemble(root: Path):
    root = Path(root)
    meta = load_committed(root, generation=0)
    if set(meta["files"]) != {"ensemble.npz"}:
        raise ContractMismatch("unexpected ensemble payload inventory")
    with np.load(root / "gen_00000000" / "ensemble.npz", allow_pickle=False) as blob:
        ensemble = blob["ensemble"]
        ids = blob["ids"].tolist()
    validate_ensemble(ensemble, ids, meta)
    return ensemble, ids, meta


def require_array_match(meta: dict, array_identity: dict) -> None:
    if meta.get("arrays") != array_identity:
        raise ContractMismatch("ensemble and target array identities differ")


def require_procedure_match(meta: dict, protocol: dict, selection: dict | None) -> None:
    procedure = meta.get("procedure", "diffusion")
    if procedure == "diffusion":
        if selection is not None and meta["checkpoint_sha256"] not in selection["test_checkpoint_hashes"]:
            raise ContractMismatch("checkpoint was not selected for test scoring")
    elif (meta.get("control") != protocol["controls"].get(procedure)
          or meta.get("training_arrays") != protocol["arrays"]["TRAIN"]):
        raise ContractMismatch("control fit or configuration differs from protocol")


def merge_ensembles(roots: list[Path], output: Path, expected_ids: list[str]) -> Path:
    if Path(output).exists():
        raise PublicationError(f"output root exists: {output}")
    rows = {}
    common = None
    chunk_hashes = {}
    for root in roots:
        ensemble, ids, meta = read_ensemble(root)
        identity = {k: v for k, v in meta.items()
                    if k not in {"ids_sha256", "files", "generation", "offset", "count"}}
        if common is None:
            common = identity
        elif identity != common:
            raise ContractMismatch("incompatible ensemble chunks")
        chunk_hashes[str(Path(root).resolve())] = sha256_file(Path(root) / "gen_00000000/COMMIT.json")
        for wid, draws in zip(ids, ensemble):
            if wid in rows:
                raise ContractMismatch(f"duplicate sampled window: {wid}")
            rows[wid] = draws
    if common is None or len(set(expected_ids)) != len(expected_ids) or set(rows) != set(expected_ids):
        raise ContractMismatch("ensemble union does not exactly cover expected windows")
    result = np.stack([rows[wid] for wid in expected_ids])
    return publish_ensemble(output, result, expected_ids, {**common, "chunks": chunk_hashes})
