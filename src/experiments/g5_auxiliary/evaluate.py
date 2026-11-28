"""Validate each candidate, freeze all six choices, then score their 2025 forecasts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.g5_auxiliary.train import EPOCHS, HIDDEN, PROTOCOL, context_features, contract
from prigm.arrays import physical_to_model
from prigm.config import N_DRAWS, UNET_WIDTHS
from prigm.loop import configure_determinism
from prigm.model import ConditionalUNet
from prigm.normalize import AffineNormalizer
from prigm.recovery import sha256_file
from prigm.run_score import decode_ensemble, hidden_primary_std
from prigm.sampler import generate_ensemble
from prigm.scoring import crps_univariate, energy_score, month_balanced_mean, order_interval
from prigm.select import select_milestone

METRICS = ("energy", "variance_crps", "variance_coverage", "variance_width")


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, allow_nan=False, indent=2) + "\n")
    os.replace(temp, path)


def score(ensemble, measured, nrm):
    prediction = decode_ensemble(ensemble, nrm)
    x = hidden_primary_std(prediction, nrm)
    y = hidden_primary_std(measured, nrm)
    variance = prediction[:, :, 5]
    obs = measured[:, 5]
    lo, hi = order_interval(variance, 2, 38)
    return [energy_score(x, y), crps_univariate(variance, obs).mean(),
            ((lo <= obs) & (obs <= hi)).mean(), (hi - lo).mean()]


def evaluate(root, arm, seed, epoch, val, holdout=None):
    configure_determinism()
    protocol = json.loads(PROTOCOL.read_text())
    if sha256_file(val) != protocol["training_inputs"]["val.npz"]:
        raise ValueError("validation digest mismatch")
    mode = "HOLDOUT2025" if holdout else "VAL"
    stem = f"{arm}_{seed}"
    folder = root / "scores" / mode / stem
    folder.mkdir(parents=True, exist_ok=True)
    checks = root / "runs" / stem / f"epoch-{epoch:02}.pt"
    ckpt_hash = sha256_file(checks)
    if holdout:
        selection = json.loads((root / "selection.json").read_text())
        if selection != choices(root):
            raise ValueError("validation choices changed")
        if selection[stem]["checkpoint"] != ckpt_hash or selection[stem]["epoch"] != epoch:
            raise ValueError("holdout restricted to selected checkpoint")
        holdout_hash = json.loads(holdout.with_suffix(".json").read_text())["arrays_sha256"]
        if sha256_file(holdout) != holdout_hash:
            raise ValueError("holdout digest mismatch")
    payload = torch.load(checks, map_location="cpu", weights_only=True)
    if (payload["contract"] != contract(arm, seed) or payload["epoch"] != epoch
            or epoch not in EPOCHS or payload["updates"] != 956 * epoch):
        raise ValueError("checkpoint identity mismatch")
    model = ConditionalUNet(n_generated=5, widths=UNET_WIDTHS)
    model.load_state_dict(payload["model"])
    model = model.cuda().eval()
    with np.load(val, allow_pickle=False) as f:
        nrm = AffineNormalizer(f["mean"], f["scale"], float(f["b_ref_sq"]), int(f["n_bins"]))
    with np.load(holdout or val, allow_pickle=False) as f:
        z, ids = f["z"], f["ids"].tolist()
    if mode == "VAL" and len(ids) != 319:
        raise ValueError("wrong validation population")
    _, obs = physical_to_model(z, nrm, 5)
    chunks = []
    for i in range(0, len(ids), 8):
        names = ids[i:i + 8]
        mask, rel = context_features(len(names), "cuda")
        draws = generate_ensemble(model, torch.from_numpy(obs[i:i + 8]).cuda(), mask, rel,
                                  names, mode, 7, 50 if holdout else 25, N_DRAWS, 5)
        chunks.append(draws)
    ensemble = np.concatenate(chunks)
    outputs = np.array([score(draws, target[HIDDEN], nrm) for draws, target in zip(ensemble, z)])
    if not np.isfinite(outputs).all():
        raise ValueError("nonfinite evaluation")
    months = np.array([name[:7] for name in ids])
    path = folder / f"epoch-{epoch:02}.npz"
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, ids=ids, draws=ensemble, metrics=outputs)
    os.replace(path.with_suffix(".tmp"), path)
    record = {"arm": arm, "seed": seed, "epoch": epoch, "checkpoint": ckpt_hash,
              "split": mode, "evaluation_source": sha256_file(Path(__file__)), "protocol": sha256_file(PROTOCOL),
              "arrays": sha256_file(holdout or val), "windows": len(ids), "archive": sha256_file(path),
              "metrics": dict(zip(METRICS, [month_balanced_mean(v, months) for v in outputs.T]))}
    write_json(path.with_suffix(".json"), record)
    print(stem, mode, epoch, record["metrics"], flush=True)


def choices(root):
    p = json.loads(PROTOCOL.read_text())
    result = {}
    for seed in p["seeds"]:
        for arm in p["arms"]:
            stem = f"{arm}_{seed}"
            rows = []
            for epoch in EPOCHS:
                source = root / "scores" / "VAL" / stem / f"epoch-{epoch:02}.json"
                row = json.loads(source.read_text())
                checkpoint = root / "runs" / stem / f"epoch-{epoch:02}.pt"
                if (row["arm"], row["seed"], row["epoch"], row["split"], row["arrays"],
                    row["protocol"], row["evaluation_source"], row["checkpoint"]) != (
                        arm, seed, epoch, "VAL", p["training_inputs"]["val.npz"],
                        sha256_file(PROTOCOL), sha256_file(Path(__file__)), sha256_file(checkpoint)):
                    raise ValueError("invalid validation record")
                rows.append({"epoch": epoch, "score": row["metrics"]["energy"], "checkpoint": row["checkpoint"]})
            result[stem] = select_milestone(rows)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["validate", "freeze", "holdout"])
    p.add_argument("root", type=Path)
    p.add_argument("--val", type=Path)
    p.add_argument("--arm", choices=["G5", "G5aux"])
    p.add_argument("--seed", type=int)
    p.add_argument("--epoch", type=int, choices=EPOCHS)
    p.add_argument("--holdout", type=Path)
    a = p.parse_args()
    if a.action == "freeze":
        path = a.root / "selection.json"
        selected = choices(a.root)
        if path.exists() and json.loads(path.read_text()) != selected:
            raise ValueError("refusing to replace a different selection")
        write_json(path, selected)
    else:
        if not a.val or not a.arm or a.seed is None:
            p.error("evaluation needs --val, --arm, and --seed")
        if a.action == "holdout" and not a.holdout:
            p.error("holdout evaluation needs --holdout")
        if a.action == "holdout":
            a.epoch = json.loads((a.root / "selection.json").read_text())[f"{a.arm}_{a.seed}"]["epoch"]
        elif a.epoch is None or a.holdout is not None:
            p.error("validation needs --epoch and validation data only")
        evaluate(a.root, a.arm, a.seed, a.epoch, a.val, a.holdout)
