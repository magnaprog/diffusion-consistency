"""Evaluate the final training epoch on the same noise as the registered 2025 forecasts."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from experiments.g5_auxiliary.evaluate import choices, score
from experiments.g5_auxiliary.train import EPOCHS, HIDDEN, PROTOCOL, context_features, contract
from prigm.arrays import physical_to_model
from prigm.config import N_DRAWS, UNET_WIDTHS
from prigm.loop import configure_determinism
from prigm.model import ConditionalUNet
from prigm.normalize import AffineNormalizer
from prigm.recovery import atomic_write_bytes, sha256_file
from prigm.sampler import generate_ensemble


def evaluate_final(root, arm, seed, val, holdout, output):
    epoch = EPOCHS[-1]
    key = f"{arm}_{seed}"
    registered = json.loads((root / "results.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    if selection != choices(root) or registered["protocol_sha256"] != sha256_file(PROTOCOL):
        raise ValueError("unrecognized validation selection or protocol")
    selected_epoch = selection[key]["epoch"]
    selected = root / "scores" / "HOLDOUT2025" / key / f"epoch-{selected_epoch:02}.json"
    previous = json.loads(selected.read_text())
    if (previous["arrays"] != sha256_file(holdout)
            or previous["archive"] != registered["fits"][key]["forecast_archive"]
            or sha256_file(val) != json.loads(PROTOCOL.read_text())["training_inputs"]["val.npz"]):
        raise ValueError("inputs differ from the registered study")

    checkpoint = root / "runs" / key / f"epoch-{epoch:02}.pt"
    validation = root / "scores" / "VAL" / key / f"epoch-{epoch:02}.json"
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != json.loads(validation.read_text())["checkpoint"]:
        raise ValueError("checkpoint differs from the final validation candidate")
    configure_determinism()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["contract"] != contract(arm, seed) or state["epoch"] != epoch or state["updates"] != 956 * epoch:
        raise ValueError("unexpected fit or training duration")
    model = ConditionalUNet(n_generated=5, widths=UNET_WIDTHS)
    model.load_state_dict(state["model"])
    model = model.cuda().eval()
    with np.load(val) as f:
        nrm = AffineNormalizer(f["mean"], f["scale"], float(f["b_ref_sq"]), int(f["n_bins"]))
    with np.load(holdout) as f:
        truth, ids = f["z"], f["ids"]
    _, obs = physical_to_model(truth, nrm, 5)
    chunks = []
    for i in range(0, len(ids), 8):
        names = ids[i:i + 8].tolist()
        mask, relative = context_features(len(names), "cuda")
        chunks.append(generate_ensemble(model, torch.from_numpy(obs[i:i + 8]).cuda(), mask,
                                        relative, names, "HOLDOUT2025", 7, 50, N_DRAWS, 5))
    draws = np.concatenate(chunks)
    metrics = np.array([score(x, y[HIDDEN], nrm) for x, y in zip(draws, truth)])
    if not np.isfinite(metrics).all():
        raise ValueError("nonfinite forecast score")
    path = output / f"{key}.npz"
    if path.exists():
        raise FileExistsError(path)
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, ids=ids, draws=draws, metrics=metrics,
                            checkpoint=checkpoint_sha, arrays=previous["arrays"], epoch=epoch)
        atomic_write_bytes(path, buffer.getvalue())
    print(key, epoch, sha256_file(path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("arm", choices=("G5", "G5aux"))
    parser.add_argument("seed", type=int, choices=(71, 73, 79))
    parser.add_argument("val", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("output", type=Path)
    evaluate_final(**vars(parser.parse_args()))
