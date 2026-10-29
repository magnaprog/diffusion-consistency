"""Independent reductions for the October camera-ready diagnostics.

The public records contain per-window statistics, not full neural ensembles.
Their upstream payload pins bind the separate server replay and rank audit.
"""
from pathlib import Path
import re

import numpy as np


CHANNELS = ["mu_x", "mu_y", "mu_z", "m", "v_s"]
METHODS = ("D", "I-S", "O-S", "G5", "gaussian", "full_flank_gaussian")
SEEDS = [7, 13, 42]
FIGURES = {"figures/calibration/fig_calibration.pdf", "figures/checkpoints/fig_checkpoints.pdf",
           "figures/overview/fig_overview.pdf"}
CALIBRATION_PINS = {
    "summary.json": "ad86c8344ee3c84d8423d0818a1d06ab7b0964ef70a7289bcc6ffcd32461873d",
    "per_window.npz": "d8f738012e3c8d2e977e21c77d404c5e7a76c2f996202127be02011c57ee65d4",
    "PROTOCOL.json": "00f15fae93bbb01846ec53ab726815464a28c7708d9ee79109ead0e7363899b5",
}
FIGURE_PINS = {
    "fig_overview": "b526f9b0270378785388a6fdbbe1b86e96039d54f7ab29bd6a0cb69e86e2d0ec",
    "fig_calibration": "aa39c366feedf5770309a8515fba0eacfc8d884d3cd51a41b947783ba7c9a4af",
    "fig_checkpoints": "8c56aaac2b5fb2e1a0c3525b909884b01ebbd0efe334894b92395d887390fe2f",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and actual.dtype.kind in "iuf"
            and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=0, atol=2e-13), label + " differs")


def tail_reductions(record, arrays, ids):
    """Recompute both tail intervals by indexing sampled years directly."""
    require(record["channels"] == CHANNELS and record["outcomes"] == ["below", "inside", "above"]
            and record["n_windows"] == 480 and record["n_draws"] == 39
            and record["interval_order_statistics"] == [2, 38]
            and record["continuous_exchangeability_reference"] == [.05, .9, .05], "tail estimand")
    require(record["bootstrap"] == {"replicates": 10000, "seed": 7003, "schemes": ["year", "twoyear"]},
            "tail bootstrap settings")
    require(set(arrays) == {*METHODS, "ids", "months"} and set(record["methods"]) == set(METHODS),
            "complete tail inventory")
    require(arrays["ids"].tolist() == ids and len(ids) == 480, "tail window order")
    months = np.array([int(w[:4]) * 100 + int(w[5:7]) for w in ids])
    require(np.array_equal(arrays["months"], months), "tail month labels")
    unique = np.unique(months)
    require(unique.tolist() == [y * 100 + m for y in range(2005, 2025) for m in (8, 9)],
            "complete TEST months")
    rng = np.random.Generator(np.random.PCG64(7003))
    draws = {"year": rng.integers(0, 20, (10000, 20))}
    blocks = rng.integers(0, 10, (10000, 10))
    draws["twoyear"] = np.stack([2 * blocks, 2 * blocks + 1], axis=-1).reshape(10000, 20)
    for name in METHODS:
        row, values = record["methods"][name], arrays[name]
        seeds = [None] if name in ("gaussian", "full_flank_gaussian") else SEEDS
        require(row["fixed_model_seeds"] == seeds, "tail model order: " + name)
        require(values.shape == (len(seeds), 480, 5, 3) and values.dtype.kind == "f"
                and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
                "tail fraction population: " + name)
        close(values.sum(axis=-1), np.ones(values.shape[:-1]), "tail partition: " + name)
        close(12 * values, np.rint(12 * values), "twelve-position counts: " + name)
        by_month = np.stack([values[:, months == m].mean(axis=1) for m in unique], axis=1)
        models = by_month.mean(axis=1)
        mean = models.mean(axis=0)
        close(row["model_means"], models, "model tail means: " + name)
        close(row["mean"], mean, "mean tails: " + name)
        close(row["below_minus_above"], mean[:, 0] - mean[:, 2], "tail imbalance: " + name)
        by_year = by_month.mean(axis=0).reshape(20, 2, 5, 3).mean(axis=1)
        for scheme, indices in draws.items():
            boot = by_year[indices].mean(axis=1)
            close(row[scheme + "_95"], np.quantile(boot, [.025, .975], axis=0),
                  scheme + " tail intervals: " + name)
            close(row[scheme + "_below_minus_above_95"],
                  np.quantile(boot[..., 0] - boot[..., 2], [.025, .975], axis=0),
                  scheme + " paired imbalance intervals: " + name)
    return record["methods"]


def claimed(paper, tex):
    return (Path(paper, "evidence/followup").exists()
            or any(name in tex for name in ("fig:calibration", "fig:checkpoints", "app:checkpoints")))


def validate_calibration(paper, real):
    from experiments import verify_prigm_paper as g
    root = g.local(paper, "evidence/followup/calibration")
    manifest = g.read_json(root / "MANIFEST.json")
    g.private_check(manifest)
    require(set(manifest) == {"schema_version", "files", "original_summary_sha256", "protocol_sha256",
                             "producer_sha256", "source", "transformation"}, "calibration manifest fields")
    require(manifest["schema_version"] == 1, "calibration manifest version")
    require(manifest["original_summary_sha256"] ==
            "c108030cdaf5d93b98c913ea85040516c742c60dbe06df37fac3356ad659caca", "original calibration report")
    require(manifest["transformation"] ==
            "Only summary.input_sha256 keys: replace the original-study root prefix with original-study/ "
            "and the accepted-evidence root prefix with accepted-evidence/. All other parsed summary "
            "content is unchanged. per_window.npz and PROTOCOL.json are byte-identical copies.",
            "calibration export transformation")
    require(set(manifest["files"]) == {"summary.json", *CALIBRATION_PINS}, "calibration payload inventory")
    g.deadline_inventory(root, {"MANIFEST.json", *manifest["files"]})
    for name, pin in manifest["files"].items():
        g.agree(g.digest(g.local(root, name)), pin, "calibration payload " + name)
    for name, pin in CALIBRATION_PINS.items():
        g.agree(manifest["files"][name], pin, "independent calibration pin " + name)
    report = g.read_json(root / "summary.json")
    g.private_check(report)
    plan = g.read_json(root / "PROTOCOL.json")
    g.private_check(plan)
    producer = "analysis/prigm_tail_calibration.py"
    source = "analysis/export_prigm_calibration.py"
    g.agree(manifest["source"], {"path": Path(source).name, "sha256": g.documentation.source_hash(g.REPO, source)},
            "calibration exporter")
    g.agree(manifest["producer_sha256"], g.documentation.source_hash(g.REPO, producer), "calibration producer")
    g.agree(plan["new_analysis_sha256"], manifest["producer_sha256"], "calibration protocol producer")
    g.agree(manifest["protocol_sha256"], CALIBRATION_PINS["PROTOCOL.json"], "calibration protocol")
    g.agree(report["per_window_sha256"], CALIBRATION_PINS["per_window.npz"], "tail array binding")
    expected_sources = {p.relative_to(g.REPO).as_posix(): g.documentation.source_hash(g.REPO, p.relative_to(g.REPO).as_posix())
                        for p in [g.REPO / producer, *sorted((g.REPO / "prigm").glob("*.py"))]}
    g.agree(report["source_sha256"], expected_sources, "complete calibration source inventory")
    real_manifest = g.read_json(g.local(paper, "evidence/real/MANIFEST.json"))
    deadline = g.read_json(g.local(paper, "evidence/deadline/MANIFEST.json"))
    expected_inputs = {"original-study/artifacts/inventory/test.npz":
                       real_manifest["input_sha256"]["artifacts/inventory/test.npz"]}
    for group, parent in (("real", real_manifest), ("deadline", deadline)):
        expected_inputs[f"accepted-evidence/{group}/MANIFEST.json"] = g.digest(
            g.local(paper, f"evidence/{group}/MANIFEST.json"))
    for name in ("comparison.json", "diagnostics.json"):
        expected_inputs["accepted-evidence/real/" + name] = real_manifest["output_sha256"][name]
    for name in ("D", "I-S", "O-S", "G5", "gaussian"):
        for seed in ([None] if name == "gaussian" else SEEDS):
            generation = None if seed is None else real[0]["methods"][name + "_raw"]["generations"][str(seed)]
            prefix = ("artifacts/controls_test/gaussian" if seed is None else
                      f"artifacts/test/{name}_s{seed}-g{generation}/ensemble")
            for filename in ("COMMIT.json", "ensemble.npz"):
                key = prefix + "/gen_00000000/" + filename
                expected_inputs["original-study/" + key] = real_manifest["input_sha256"][key]
    for name in ("gaussian/test/SCORES.json", "gaussian/test/lambda_0.1_rho_0/gen_00000000/COMMIT.json",
                 "gaussian/test/lambda_0.1_rho_0/gen_00000000/ensemble.npz"):
        expected_inputs["accepted-evidence/deadline/" + name] = deadline["files"][name]
    g.agree(report["input_sha256"], expected_inputs, "complete calibration input inventory")
    with np.load(root / "per_window.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    tails = tail_reductions(report, arrays, real[0]["ids"])
    diagnostics = g.read_json(g.local(paper, "evidence/real/diagnostics.json"))
    for name in ("D", "I-S", "O-S", "G5", "gaussian"):
        original = diagnostics["methods"][name + "_raw"]["metrics"]["cov90_by_position_channel"]["month_balanced"]
        close(np.asarray(tails[name]["mean"])[:, 1], np.mean(original, axis=0), "accepted coverage: " + name)
    return report


def figure_record(paper, name, inputs):
    from experiments import verify_prigm_paper as g
    folder = {"fig_calibration": "calibration", "fig_checkpoints": "checkpoints", "fig_overview": "overview"}[name]
    root = g.local(paper, "figures/" + folder)
    path = root / (name + ".manifest.json")
    require(name in FIGURE_PINS and g.digest(path) == FIGURE_PINS[name], "reviewed figure manifest: " + name)
    record = g.read_json(path)
    g.private_check(record)
    g.agree(record["source"], {"path": name + ".py", "sha256": g.digest(g.local(paper, name + ".py"))},
            "figure source " + name)
    g.agree(record.get("style"), {"path": "prigm.mplstyle",
                                  "sha256": g.digest(g.local(paper, "prigm.mplstyle"))},
            "figure style " + name)
    g.agree(record["inputs"], {k: {"path": Path(v).name if name == "fig_calibration" else str(v),
                                  "sha256": g.digest(g.local(paper, v))}
                               for k, v in inputs.items()}, "figure inputs " + name)
    outputs = {name + ".png", name + ".pdf"}
    require(set(record["output_sha256"]) == outputs, "figure output inventory")
    g.deadline_inventory(root, {*outputs, name + ".manifest.json"})
    for filename, digest in record["output_sha256"].items():
        g.agree(g.digest(root / filename), digest, "figure output " + filename)
    return record


def validate_calibration_figure(paper, report):
    from experiments import verify_prigm_paper as g
    op = g.read_json(g.local(paper, "evidence/deadline/operator-mechanism.json"))
    figure = figure_record(paper, "fig_calibration", {
        "operator": "evidence/deadline/operator-mechanism.json",
        "calibration": "evidence/followup/calibration/summary.json"})
    terms = ("target_term", "dispersion_term", "energy_score")
    deltas = np.array([[op["methods"]["G5"]["seedwise"][str(s)]["metrics"]["raw_" + t]["month_balanced"]
                        - op["methods"]["D"]["seedwise"][str(s)]["metrics"]["raw_" + t]["month_balanced"]
                        for t in terms] for s in SEEDS])
    deltas = np.vstack([deltas, deltas.mean(axis=0)])
    require([r["seed"] for r in figure["decomposition"]] == [*SEEDS, "mean"], "plotted decomposition order")
    close([[r["delta_target"], -r["negative_delta_dispersion"], r["delta_energy"]]
           for r in figure["decomposition"]], deltas, "plotted score contributions")
    require(figure["method_order"] == ["D", "O-S", "G5", "full_flank_gaussian"]
            and figure["channels"] == CHANNELS and figure["outcomes"] == ["below", "inside", "above"],
            "plotted tail order")
    for name in figure["method_order"]:
        expected, actual = report["methods"][name], figure["tails"][name]
        require(actual["fixed_model_seeds"] == expected["fixed_model_seeds"], "plotted seed order")
        for key in ("mean", "model_means", "year_95"):
            close(actual[key + "_percent"], 100 * np.asarray(expected[key]), "plotted tail percentages")
    require(figure["pointwise_intervals_plotted"] is False, "fixed seed dots cannot claim confidence intervals")
    return figure


def calibration_claims(tex, rendered, report, figure):
    from experiments import verify_prigm_paper as g
    # Normalize this prose count only; interval commas remain significant.
    rendered = re.sub(r"\b10,000(?=\s+resamples)", "10000", rendered)
    floats = tex.replace(r"\begin{figure}", r"\begin{table}").replace(r"\end{figure}", r"\end{table}")
    block = g.float_block(floats, "fig:calibration")
    require(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block)
            == ["figures/calibration/fig_calibration.pdf"], "calibration label and image differ")
    caption = ("Why scores and interval coverage differ. Diffusion results use the checkpoints that minimize raw validation score. "
        "(a) G5 minus D for each seed and their mean. The change in distance to the withheld record, less the change in the spread term, gives the energy score difference. "
        "Both terms use 60 coordinates standardized on TRAIN; spread is half the mean distance between distinct predictions. "
        "(b,c) Fractions of observations below and above nominal 90\\% intervals for the five primary quantities. "
        "Bars average three diffusion fits or show one Gaussian ensemble. Gray dots show seeds 7, 13 and 42 from left to right. "
        "The dashed 5\\% line assumes that, at each time and coordinate, the observation and ensemble members are exchangeable, with no ties. Months receive equal weight. "
        "This analysis uses the existing TEST ensembles after the original results were available.")
    require(g.prose_text(g.float_caption(floats, "fig:calibration"), source=True)
            == g.prose_text(caption, source=True), "calibration caption differs")
    def bind(anchor, clause):
        g.bind_clause((tex, rendered), g.clause_pattern(g.prose_text(anchor, source=True)), clause)
    bind("Why scores and interval coverage differ", caption)
    rows = {row["seed"]: row for row in figure["decomposition"]}
    r13, r42 = rows[13], rows[42]
    require(r13["delta_target"] < 0 < r13["negative_delta_dispersion"]
            and r42["negative_delta_dispersion"] < 0 < r42["delta_target"]
            and max(r13["delta_energy"], r42["delta_energy"]) < 0, "seedwise contribution directions")
    # Locate the subject independently of the seed value checked in the clause.
    for subject, clause in (
        ("G5", "At seed 13, G5 reduces both distance to the measurements and spread, with the larger decrease in distance lowering the score."),
        ("both terms", "At seed 42, both terms increase, and the larger increase in spread lowers the score."),
    ):
        g.bind_clause((tex, rendered), g.clause_pattern("at seed") + r"\s+[^,\s]+\s*,\s*" + g.clause_pattern(subject), clause)
    bind("These are arithmetic explanations", "These are arithmetic explanations of the existing predictions. G5 changes both the generated representation and the training loss, so their separate effects on learning remain unresolved.")
    methods = report["methods"]
    means = {name: 100 * np.asarray(row["mean"])[4] for name, row in methods.items()}
    for name in ("D", "G5"):
        directions = np.asarray(methods[name]["model_means"])[:, 4, 0] - np.asarray(methods[name]["model_means"])[:, 4, 2]
        require(np.array_equal(np.sign(directions), [1, 1, -1]), "seedwise tail reversal: " + name)
    bind("G5's nominal", rf"G5's nominal 90\% intervals cover only {means['G5'][1]:.1f}\% of the $v_s$ observations.")
    g.bind_clause((tex, rendered), g.clause_pattern("across the") + r"\s+[^\s]+\s+" + g.clause_pattern("fits,"),
         rf"Across the three fits, {means['G5'][0]:.1f}\% of observations fall below G5's $v_s$ intervals and {means['G5'][2]:.1f}\% above them.")
    bind("shows why the mean alone", "shows why the mean alone is incomplete: errors are predominantly below the intervals at seeds 7 and 13, but above them at seed 42. D has the same reversal at seed 42.")
    bind("For O-S, the mean fractions", rf"For O-S, the mean fractions are {means['O-S'][0]:.1f}\% below and {means['O-S'][2]:.1f}\% above.")
    ci = 100 * np.asarray(methods["G5"]["year_below_minus_above_95"])[:, 4]
    contrast = 100 * methods["G5"]["below_minus_above"][4]
    bind(r"For G5's $v_s$ intervals, the fraction below",
         rf"For G5's $v_s$ intervals, the fraction below exceeds the fraction above by {contrast:.1f} percentage points. "
         rf"The 95\% interval is $[{ci[0]:.1f},{ci[1]:.1f}]$, using 10{{,}}000 resamples of years and the same years for both fractions. "
         "Repeated occurrences of each year retain their weight. The difference uses percentages before rounding. "
         "This contrast quantifies temporal variation for the existing models, checkpoint choices and samples; new fits or new checkpoint choices would add other sources of uncertainty.")
    bind("Gaussian regression covers", rf"Gaussian regression covers {means['full_flank_gaussian'][1]:.1f}\% of $v_s$ observations, with "
         rf"{means['full_flank_gaussian'][0]:.1f}\% below and {means['full_flank_gaussian'][2]:.1f}\% above its intervals.")


def checkpoint_claims(tex, rendered, checked):
    """Bind the reported choices, contrasts and resampling scope to checked data."""
    from experiments import verify_prigm_paper as g
    from experiments._prigm_checkpoint_evidence import UPDATES
    summary, saved = checked["summary"], checked["saved"]
    # Normalize grouped counts only in their stated prose contexts.
    rendered = re.sub(r"\b10,000(?=\s+resamples)|(?<=update )9,560(?=\s+in)",
                      lambda m: m[0].replace(",", ""), rendered)
    policies = summary["policies"]
    require(len(policies) == 6 and all(p["P25"] == p["P50"] for p in policies.values()),
            "unchanged six validation choices")
    def bind(anchor, clause):
        g.bind_clause((tex, rendered), g.clause_pattern(g.prose_text(anchor, source=True)), clause)
    g.bind_clause((tex, rendered), r"all\s+[^\s]+\s+" + g.clause_pattern("validation choices"),
                  "All six validation choices agree between 25 and 50 evaluations for these fits and initial noise.")
    deltas = summary["common_update_TEST_contrasts"]
    values = [f"${deltas[str(u)]['mean']:.4f}$" for u in UPDATES]
    g.bind_clause((tex, rendered), g.clause_pattern("at the") + r"\s+[^\s]+\s+" + g.clause_pattern("common update counts, mean TEST"), "at the five common update counts, mean TEST differences for O-S minus D are "
         + ", ".join(values[:-1]) + ", and " + values[-1] + ".")
    signs = [np.sign(deltas[str(u)]["seedwise"]) for u in UPDATES]
    require(all(np.all(s == -1) for s in signs[-2:])
            and all(set(s) == {-1, 1} for s in signs[:3]), "common-update seed directions")
    bind("O-S has lower scores at", "O-S has lower scores at all three seeds at the last two counts; seed differences have mixed signs at the first three.")
    changes = summary["final_minus_selected_TEST"]["P25"]
    means = summary["policy_means"]
    bind("Moving all fits", "Moving all fits to the final update raises D's mean TEST score by "
         f"{np.mean(changes['D']):.4f} and O-S's by {np.mean(changes['O-S']):.4f}, changing O-S minus D from "
         f"${means['P25']['O-S_minus_D']:.4f}$ to ${means['F']['O-S_minus_D']:.4f}$.")
    frequencies = [record["D_s7"]["bootstrap_selection_counts"]["9560"] / 100
                   for record in (saved["curves"], summary["VAL50_selection_diagnostics"])]
    g.bind_clause((tex, rendered), r"in\s+[^\s]+\s+" + g.clause_pattern("resamples of the"),
         "In 10{,}000 resamples of the twenty years, D at seed 7 chooses "
         rf"update 9{{,}}560 in {frequencies[0]:.1f}\% of resamples with 25 evaluations and {frequencies[1]:.1f}\% with 50; "
         "the other resamples choose the final update.")
    for rows in (saved["curves"], summary["VAL50_selection_diagnostics"]):
        for name in ("D_s13", "D_s42"):
            require(rows[name]["bootstrap_selection_counts"][str(policies[name]["P25"])] == 10000,
                    "D earlier-choice stability")
    g.bind_clause((tex, rendered), g.clause_pattern("D at seeds") + r"\s+[^\s]+\s+and\s+[^\s]+\s+" + g.clause_pattern("retains"),
                  "D at seeds 13 and 42 retains its respective earlier choice in every resample under either evaluation count.")
    for anchor, clause in (
        ("validation ensembles with", "validation ensembles with 50 evaluations reuse the initial noise from the original ensembles with 25. Their validation scores determine the model choices before additional TEST samples are generated."),
        ("Establishing whether regularization", "Establishing whether regularization or overfitting causes this pattern is a separate empirical question."),
        ("Each sampled year contributes", "Each sampled year contributes both validation months, including repeated occurrences, and all checkpoints use the same resampled years."),
        ("With weights and samples", "With weights and samples held constant, these frequencies measure sensitivity to the years represented in validation."),
    ):
        bind(anchor, clause)
    floats = tex.replace(r"\begin{figure}", r"\begin{table}").replace(r"\end{figure}", r"\end{table}")
    block = g.float_block(floats, "fig:checkpoints")
    require(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block)
            == ["figures/checkpoints/fig_checkpoints.pdf"], "checkpoint label and image differ")
    caption = ("How the D/O-S comparison changes during training. The subplots show VAL with 25 DDIM evaluations, VAL with 50, and TEST with 50. "
        "Each point gives equal weight to months; ensembles contain 39 samples. Colors and open or filled markers distinguish D and O-S, while line styles and marker shapes distinguish seeds. "
        "Black rings mark the validation choice with 25 evaluations, P25; crosses mark the choice with 50, P50. Both choices coincide for all six fits. "
        "F marks the common final update. Validation and test scores concern different windows, and model choice precedes the additional TEST analysis.")
    require(g.prose_text(g.float_caption(floats, "fig:checkpoints"), source=True) == g.prose_text(caption, source=True),
            "checkpoint caption differs")
    bind("How the D/O-S comparison changes", caption)


def validate_reselection(paper, checked):
    """Recompute the paired-year sensitivity from authenticated window scores.

    Draw year indices directly, preserving repeated years. This does not use the
    producer's month-reduction, selection or bootstrap helpers, or retrain fits.
    """
    from experiments import verify_prigm_paper as g
    from experiments import _prigm_checkpoint_evidence as c
    record = g.read_json(g.local(paper, "evidence/followup/checkpoint_reselection.json"))
    g.private_check(record)
    summary = checked["summary"]
    input_path = "evidence/followup/checkpoints/SUMMARY.json"
    input_hash = g.digest(g.local(paper, input_path))
    g.agree(input_hash, checked["manifest"]["files"]["SUMMARY.json"], "reselection authenticated scores")
    g.agree(g.read_json(g.local(paper, input_path)), summary, "reselection supplied scores")
    years = {}
    for split in ("VAL", "TEST"):
        data = summary["scores"][split]
        months = c.window_months(data["ids"], data["months"], split)
        g.agree(sorted(data["data"]), sorted(c.JOBS), "reselection complete grid")
        windows = np.asarray([data["data"][job] for job in c.JOBS])
        require(windows.shape == (30, c.COUNTS[split]) and windows.dtype.kind in "if"
                and np.isfinite(windows).all(), "reselection finite score population")
        monthly = np.stack([windows[:, months == m].mean(axis=1) for m in np.unique(months)], axis=-1)
        years[split] = monthly.reshape(6, 5, 20, 2).mean(axis=-1)
    original = years["VAL"].mean(axis=-1).argmin(axis=1)
    choices = {name: c.UPDATES[index] for name, index in zip(c.NAMES, original)}
    g.agree(choices, {name: summary["policies"][name]["P50"] for name in c.NAMES},
            "reselection original VAL choices")
    sampled = np.random.Generator(np.random.PCG64(7003)).integers(0, 20, size=(10000, 20))
    # Shape: fit, checkpoint, replicate. Identical indices pair all comparisons.
    validation = years["VAL"][:, :, sampled].mean(axis=-1)
    test = years["TEST"][:, :, sampled].mean(axis=-1)
    winners = validation.argmin(axis=1)  # Ascending updates resolve exact ties.
    fixed = test[np.arange(6), original]
    reselected = np.take_along_axis(test, winners[:, None, :], axis=1)[:, 0, :]
    observed = years["TEST"].mean(axis=-1)[np.arange(6), original]
    expected = {
        "schema_version": 1, "analysis": "checkpoint_reselection_existing_scores",
        "scope": "Exploratory sensitivity conditional on existing fits, saved checkpoints and ensemble draws; previously exposed TEST period.",
        "limits": ["No retraining or new model samples", "No new-fit uncertainty",
                   "No nominal coverage claim under temporal dependence",
                   "Accepted fixed-selection intervals and historical evidence unchanged"],
        "procedure": {"generator": "PCG64", "seed": 7003, "replicates": 10000,
                      "years": list(range(2005, 2025)), "validation_evaluations": 50,
                      "test_evaluations": 50, "ensemble_members": 39,
                      "updates": list(c.UPDATES), "groups": list(c.NAMES),
                      "weighting": "Equal represented-month means; repeated year occurrences retain multiplicity",
                      "pairing": "Identical year multiplicities for VAL and TEST, all groups, fixed and reselected comparisons",
                      "selection": "VAL only; exact ties to earlier update; all three fits retained"},
        "original_choices": choices,
        "observed_TEST": {"D": float(observed[:3].mean()), "O-S": float(observed[3:].mean()),
                          "O-S_minus_D": float(observed[3:].mean() - observed[:3].mean())},
        "selection_counts": {name: {str(update): int(np.count_nonzero(winners[i] == j))
                                    for j, update in enumerate(c.UPDATES)} for i, name in enumerate(c.NAMES)},
        "input": {"path": "paper/workshops/prigm/" + input_path, "sha256": input_hash},
        "source": {"path": "scripts/analyze_prigm_checkpoint_reselection.py",
                   "sha256": g.digest(g.local(g.REPO, "scripts/analyze_prigm_checkpoint_reselection.py"))},
        "helper_sha256": {name: g.digest(g.local(g.REPO, name))
                          for name in ("prigm/config.py", "prigm/scoring.py", "prigm/select.py")},
    }
    for name, scores in (("fixed_selection", fixed), ("reselected", reselected)):
        d, o = scores[:3].mean(axis=0), scores[3:].mean(axis=0)
        require(np.all(d > 0), "reselection positive relative denominator")
        delta = o - d
        expected[name] = {"mean": float(delta.mean()),
                          "percentile_2_5_97_5": np.quantile(delta, [.025, .975]).tolist(),
                          "relative_percentile_2_5_97_5": np.quantile(delta / d, [.025, .975]).tolist()}
    g.agree(record, expected, "checkpoint reselection")
    return record


def reselection_claims(tex, rendered, record):
    from experiments import verify_prigm_paper as g
    fixed = record["fixed_selection"]["percentile_2_5_97_5"]
    selected = record["reselected"]["percentile_2_5_97_5"]
    clause = ("Repeating checkpoint choice also tests whether this sensitivity changes the O-S/D comparison. "
              "In 10{,}000 resamples at seed 7003, the same year multiplicities weight validation and TEST. "
              "Validation with 50 evaluations chooses each fit's checkpoint anew, with ties favoring the earlier update. "
              f"The 2.5th and 97.5th percentiles of O-S minus D are $[{selected[0]:.4f},{selected[1]:.4f}]$, "
              f"close to $[{fixed[0]:.4f},{fixed[1]:.4f}]$ when the original choices are retained in the same resamples. "
              "Thus, allowing model choice to vary preserves the small aggregate difference in this exercise. "
              "Both calculations condition on the existing fits and samples; uncertainty from retraining and nominal coverage "
              "under temporal dependence remain outside this sensitivity check.")
    rendered = re.sub(r"\b\d{1,3}(?:,\d{3})+(?=\s+resamples)", lambda m: m[0].replace(",", ""), rendered)
    g.bind_clause((tex, rendered), g.clause_pattern("Repeating checkpoint choice"), clause)
    # Also count the numeric sentences independently: a false sentence outside
    # the valid paragraph must not be hidden by that complete paragraph.
    for anchor, sentence in (
        (r"in\s+[^\s]+\s+" + g.clause_pattern("resamples at seed") + r"\s+[^,\s]+\s*,\s*",
         "In 10{,}000 resamples at seed 7003, the same year multiplicities weight validation and TEST."),
        (g.clause_pattern("Validation with") + r"\s+[^\s]+\s+" + g.clause_pattern("evaluations chooses each fit"),
         "Validation with 50 evaluations chooses each fit's checkpoint anew, with ties favoring the earlier update."),
        (r"the\s+[^\s]+\s+and\s+[^\s]+\s+" + g.clause_pattern("percentiles of O-S minus D"),
         f"The 2.5th and 97.5th percentiles of O-S minus D are $[{selected[0]:.4f},{selected[1]:.4f}]$, "
         f"close to $[{fixed[0]:.4f},{fixed[1]:.4f}]$ when the original choices are retained in the same resamples."),
    ):
        g.bind_clause((tex, rendered), anchor, sentence)


def validate_overview(paper, checkpoints, calibration):
    """Bind every overview coordinate to the independently checked populations."""
    from experiments import verify_prigm_paper as g
    from experiments import _prigm_checkpoint_evidence as c
    inputs = {name: "evidence/followup/checkpoints/" + name
              for name in (*sorted(c.FILES), "MANIFEST.json")}
    inputs["calibration"] = "evidence/followup/calibration/summary.json"
    figure = figure_record(paper, "fig_overview", inputs)
    policies = checkpoints["summary"]["policies"]
    require(all(p["P25"] == p["P50"] for p in policies.values()), "overview validation choice agreement")
    # Recompute TEST month means from windows, not the producer's plotted data.
    data = checkpoints["summary"]["scores"]["TEST"]
    months = np.asarray(data["months"])
    scores = {}
    for arm in c.ARMS:
        scores[arm] = []
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            selected = []
            for policy in ("F", "P25"):
                values = np.asarray(data["data"][f"{name}-g{policies[name][policy]}"])
                selected.append(float(np.mean([values[months == m].mean() for m in np.unique(months)])))
            scores[arm].append(selected)
    contrast = np.asarray(scores["O-S"]) - scores["D"]
    g.agree(figure["panel_a"], {"policy_order": ["F", "P25"], "x": [0, 1], "seeds": SEEDS,
            "score_by_arm": scores, "seedwise": contrast.tolist(), "mean": contrast.mean(axis=0).tolist(),
            "policies": policies, "validation_choices_agree": True}, "overview checkpoint coordinates")
    methods = ["D", "O-S", "G5", "full_flank_gaussian"]
    g.agree(figure["method_order"], methods, "overview method order")
    g.agree(figure["channels"], CHANNELS, "overview channel order")
    expected = {}
    # Calibration was independently reduced from per-window fractions above.
    # Keep one Gaussian realization, without inventing three Gaussian fits.
    for index, name in enumerate(methods):
        row = calibration["methods"][name]
        seeds = [None] if name == "full_flank_gaussian" else SEEDS
        g.agree(row["fixed_model_seeds"], seeds, "overview model population")
        models = 100 * np.asarray(row["model_means"])[:, :, 1]
        require(models.shape == (len(seeds), 5), "overview model coverage shape")
        y = np.arange(5) + (index - 1.5) * .15
        expected[name] = {"fixed_model_seeds": seeds, "mean_percent": models.mean(axis=0).tolist(),
                          "model_means_percent": models.tolist(), "y": y.tolist(),
                          "individual_y": [(y + offset).tolist() for offset in (-.028, 0, .028)]
                          if len(seeds) == 3 else []}
    g.agree(figure["panel_b"], expected, "overview coverage coordinates")
    g.agree(figure["dependency_sources"], {key: {"path": name, "sha256": g.digest(g.local(paper, name))}
            for key, name in (("checkpoints", "fig_checkpoints.py"), ("calibration", "fig_calibration.py"))},
            "overview dependency sources")
    g.agree(figure["calibration_provenance"], {key: calibration[key] for key in ("bootstrap", "per_window_sha256")},
            "overview calibration provenance")
    metadata = {"schema_version": 1, "figure": "fig_overview", "protocol_sha256": c.PROTOCOL,
                "scope_sha256": c.SCOPE, "interval_order_statistics": [2, 38], "n_draws": 39, "n_windows": 480,
                "pointwise_intervals_plotted": False,
                "coverage_reference": {"percent": 90, "assumption": "at each time and coordinate, "
                                       "the observation and 39 ensemble members are exchangeable, with no ties"}}
    g.agree({key: figure[key] for key in metadata}, metadata, "overview population and reference")
    return figure


def overview_claims(tex, rendered):
    from experiments import verify_prigm_paper as g
    floats = tex.replace(r"\begin{figure}", r"\begin{table}").replace(r"\end{figure}", r"\end{table}")
    block = g.float_block(floats, "fig:overview")
    require(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block)
            == ["figures/overview/fig_overview.pdf"], "overview label and image differ")
    caption = ("Model choice and predictive uncertainty. (a) O-S minus D energy score at the final update and at the "
               "checkpoints chosen independently by raw validation score. Symbols distinguish the three training seeds; "
               "diamonds show their mean. Negative values favor O-S. Validation with 25 or 50 evaluations chooses the same six checkpoints. "
               "(b) Raw nominal 90\\% interval coverage for each "
               "magnetic quantity. Large filled markers average three diffusion fits; small open circles show individual fits. "
               "Gaussian regression uses one ensemble per window. The dashed line marks 90\\%. Scores and coverage give equal "
               "weight to months; coverage also averages the twelve hidden positions.")
    require(g.prose_text(g.float_caption(floats, "fig:overview"), source=True) == g.prose_text(caption, source=True),
            "overview caption differs")
    g.bind_clause((tex, rendered), g.clause_pattern("Model choice and predictive uncertainty"), caption)
    g.bind_clause((tex, rendered), g.clause_pattern("Validation with") + r"\s+[^\s]+\s+or\s+[^\s]+\s+"
                  + g.clause_pattern("evaluations chooses"),
                  "Validation with 25 or 50 evaluations chooses the same six checkpoints.")
