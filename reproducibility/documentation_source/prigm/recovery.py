"""Durable run-root publication (plan Section 10). Artifact commit != git commit."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class ContractMismatch(RuntimeError):
    pass


class PublicationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def publish_generation(
    run_root: Path,
    generation: int,
    payload: dict[str, bytes],
    meta: dict,
) -> Path:
    """Write payload files, then atomically publish commit manifest with hashes."""
    run_root = Path(run_root)
    gen_dir = run_root / f"gen_{generation:08d}"
    if gen_dir.exists():
        raise PublicationError(f"generation exists: {gen_dir}")
    tmp_dir = run_root / f".tmp_gen_{generation:08d}"
    if tmp_dir.exists():
        raise PublicationError(f"incomplete tmp generation: {tmp_dir}")
    tmp_dir.mkdir(parents=True)
    hashes = {}
    for name, data in payload.items():
        p = tmp_dir / name
        atomic_write_bytes(p, data)
        hashes[name] = sha256_bytes(data)
    meta = dict(meta)
    meta["generation"] = generation
    meta["files"] = hashes
    man_bytes = (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode()
    atomic_write_bytes(tmp_dir / "COMMIT.json", man_bytes)
    os.replace(tmp_dir, gen_dir)
    fsync_dir(run_root)
    latest = run_root / "LATEST"
    atomic_write_bytes(latest, f"{generation}\n".encode())
    return gen_dir


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PublicationError(f"JSON object required: {path}")
    return value


def _generations(run_root: Path) -> list[int]:
    return sorted(
        int(p.name[4:]) for p in Path(run_root).glob("gen_*")
        if p.is_dir() and p.name[4:].isdigit()
        and p.name == f"gen_{int(p.name[4:]):08d}"
    )


def _load_generation(run_root: Path, generation: int) -> dict:
    gen_dir = run_root / f"gen_{generation:08d}"
    meta = read_json(gen_dir / "COMMIT.json")
    if type(meta.get("generation")) is not int or meta["generation"] != generation:
        raise PublicationError("manifest generation mismatch")
    files = meta.get("files")
    if not isinstance(files, dict) or not files:
        raise PublicationError("manifest has no payload hashes")
    for name, expected in files.items():
        if (not isinstance(name, str) or Path(name).name != name
                or name in (".", "..", "COMMIT.json") or not isinstance(expected, str)):
            raise PublicationError("invalid manifest payload entry")
        p = gen_dir / name
        try:
            got = sha256_file(p)
        except FileNotFoundError as exc:
            raise PublicationError(f"missing payload {name}") from exc
        if got != expected:
            raise PublicationError(f"hash mismatch {name}")
    return meta


def load_committed(run_root: Path, generation: int | None = None, *, validator=None) -> dict:
    """Select newest valid commit, never trusting LATEST over directory contents.

    An explicit generation is strict. An optional validator checks payload
    semantics before automatic selection. Only PublicationError permits fallback;
    contract changes, permissions, and unrelated programming errors propagate.
    """
    run_root = Path(run_root)

    def checked(gen):
        meta = _load_generation(run_root, gen)
        if validator is not None:
            validator(meta)
        return meta

    if generation is not None:
        if type(generation) is not int or generation < 0:
            raise PublicationError("invalid requested generation")
        return checked(generation)
    try:
        hint = (run_root / "LATEST").read_text().strip()
    except FileNotFoundError:
        hint = None
    except UnicodeError:
        hint = "<invalid encoding>"
    skipped = []
    for gen in reversed(_generations(run_root)):
        try:
            meta = checked(gen)
        except PublicationError as exc:
            skipped.append({"generation": gen, "reason": str(exc)})
            continue
        if skipped or hint != str(gen):
            meta["recovery"] = {"latest_hint": hint, "selected_generation": gen,
                                "skipped": skipped, "replay_from_update": meta.get("updates", gen)}
        return meta
    raise PublicationError(f"no valid committed generation; rejected: {skipped}")


def previous_generation(run_root: Path, current: int, *, validator=None) -> int | None:
    for gen in reversed(_generations(run_root)):
        if gen >= current:
            continue
        try:
            load_committed(run_root, gen, validator=validator)
        except PublicationError:
            continue
        return gen


class RunLock:
    def __init__(self, run_root: Path):
        self.path = Path(run_root) / "WRITE.lock"
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(self.fd)
            self.fd = None
            if isinstance(e, BlockingIOError):
                raise PublicationError("run root locked by another writer") from e
            raise
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


PHASE_TEST_STARTED = "TEST_STARTED"


def publish_phase(run_root: Path, phase: str, payload: dict) -> None:
    data = json.dumps({"phase": phase, **payload}, sort_keys=True, indent=2).encode()
    atomic_write_bytes(Path(run_root) / "PHASE.json", data)


def read_phase(run_root: Path) -> dict | None:
    p = Path(run_root) / "PHASE.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())
