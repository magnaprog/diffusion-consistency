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
FIGURES = {"figures/calibration/fig_calibration.pdf", "figures/checkpoints/fig_checkpoints.pdf"}
CALIBRATION_PINS = {
    "summary.json": "ad86c8344ee3c84d8423d0818a1d06ab7b0964ef70a7289bcc6ffcd32461873d",
    "per_window.npz": "d8f738012e3c8d2e977e21c77d404c5e7a76c2f996202127be02011c57ee65d4",
    "PROTOCOL.json": "00f15fae93bbb01846ec53ab726815464a28c7708d9ee79109ead0e7363899b5",
}
FIGURE_PINS = {
    "fig_calibration": "ad2af45c5157435c902fb49e7c9e0950567692e6abf4286d6fa6c5824a44ec91",
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
    folder = "calibration" if name == "fig_calibration" else "checkpoints"
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
    g.agree(record["inputs"], {k: {"path": str(v) if name == "fig_checkpoints" else Path(v).name,
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
    caption = ("Score contributions and interval errors for the original ensembles, using diffusion checkpoints selected by raw validation score. "
        "(a) G5 minus D for each seed and their mean. The change in distance to withheld measurements minus the change in the spread term equals the difference in energy score. "
        "Both terms use 60 coordinates standardized on TRAIN; the spread term is half the mean Euclidean distance between distinct draws. "
        "(b,c) Fractions of observations below and above nominal 90\\% intervals for the five primary coordinates. "
        "Bars average three diffusion fits or show one Gaussian realization. Gray dots within each diffusion bar give seeds 7, 13 and 42 from left to right. "
        "The dashed 5\\% reference assumes continuous exchangeability of the observation and ensemble members. Months receive equal weight. "
        "These comparisons use the existing TEST ensembles and were made after the original results were available.")
    require(g.prose_text(g.float_caption(floats, "fig:calibration"), source=True)
            == g.prose_text(caption, source=True), "calibration caption differs")
    def bind(anchor, clause):
        g.bind_clause((tex, rendered), g.clause_pattern(g.prose_text(anchor, source=True)), clause)
    bind("Score contributions and interval errors", caption)
    rows = {row["seed"]: row for row in figure["decomposition"]}
    r13, r42 = rows[13], rows[42]
    require(r13["delta_target"] < 0 < r13["negative_delta_dispersion"]
            and r42["negative_delta_dispersion"] < 0 < r42["delta_target"]
            and max(r13["delta_energy"], r42["delta_energy"]) < 0, "seedwise contribution directions")
    # Exact seed coordinates remain checked by validate_calibration_figure.
    bind("At seed 13, G5", "At seed 13, G5 decreases both distance to the measurements and spread, with the larger decrease in distance.")
    bind("At seed 42, both", "At seed 42, both increase, with the larger increase in spread. Both patterns lower the score.")
    bind("This decomposition explains", "This decomposition explains the score differences for the existing predictions. It cannot identify a training mechanism or separate G5's changes to representation and loss.")
    methods = report["methods"]
    means = {name: 100 * np.asarray(row["mean"])[4] for name, row in methods.items()}
    for name in ("D", "G5"):
        directions = np.asarray(methods[name]["model_means"])[:, 4, 0] - np.asarray(methods[name]["model_means"])[:, 4, 2]
        require(np.array_equal(np.sign(directions), [1, 1, -1]), "seedwise tail reversal: " + name)
    bind("G5's nominal", rf"G5's nominal 90\% intervals still miss many observations, especially for $v_s$, whose coverage is {means['G5'][1]:.1f}\%.")
    # Anchor before the percentage so malformed values cannot hide beside a
    # valid duplicate. The fraction-difference clause has a separate anchor.
    g.bind_clause((tex, rendered), g.clause_pattern("averaged over the three fits,"),
         rf"Averaged over the three fits, {means['G5'][0]:.1f}\% of observations lie below G5's $v_s$ intervals and {means['G5'][2]:.1f}\% above them.")
    bind("show a reversal that the averages obscure", "show a reversal that the averages obscure: errors lie predominantly below the intervals at seeds 7 and 13, and above them at seed 42. D also reverses direction at seed 42.")
    bind("O-S's mean fractions", rf"O-S's mean fractions are {means['O-S'][0]:.1f}\% below and {means['O-S'][2]:.1f}\% above.")
    ci = 100 * np.asarray(methods["G5"]["year_below_minus_above_95"])[:, 4]
    contrast = 100 * methods["G5"]["below_minus_above"][4]
    bind(r"For G5's $v_s$ intervals, the fraction below", rf"For G5's $v_s$ intervals, the fraction below exceeds the fraction above by {contrast:.1f} percentage points, with a 95\% interval of $[{ci[0]:.1f},{ci[1]:.1f}]$ from resampling years.")
    g.bind_clause((tex, rendered), g.clause_pattern("gaussian regression covers")
         + r"(?!\s+" + g.clause_pattern("all twelve") + r"\b)",
         rf"Gaussian regression covers {means['full_flank_gaussian'][1]:.1f}\% of $v_s$ observations, with "
         rf"{means['full_flank_gaussian'][0]:.1f}\% below and {means['full_flank_gaussian'][2]:.1f}\% above its intervals.")
    for clause in (
        "The difference is calculated before rounding the two tail percentages.",
        "Each of 10{,}000 resamples calculates both fractions on the same sampled years and retains every occurrence of a repeated year.",
        "The interval concerns this one contrast, conditional on the existing models, checkpoints and draws; it excludes variation from new training seeds or a new checkpoint choice.",
    ):
        bind(" ".join(clause.split()[:6]), clause)


def checkpoint_claims(tex, rendered, checked):
    """Bind the reported policies, contrasts and resampling scope to checked data."""
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
    bind("All six validation choices", "All six validation choices agree between 25 and 50 evaluations.")
    deltas = summary["common_update_TEST_contrasts"]
    values = [f"${deltas[str(u)]['mean']:.4f}$" for u in UPDATES]
    bind("At the five common update counts, mean TEST", "At the five common update counts, mean TEST differences for O-S minus D are "
         + ", ".join(values[:-1]) + ", and " + values[-1] + ", respectively.")
    signs = [np.sign(deltas[str(u)]["seedwise"]) for u in UPDATES]
    require(all(np.all(s == -1) for s in signs[-2:])
            and all(set(s) == {-1, 1} for s in signs[:3]), "common-update seed directions")
    bind("O-S has lower scores at", "O-S has lower scores at all three seeds at the last two counts; seed differences have mixed signs at the first three.")
    changes = summary["final_minus_selected_TEST"]["P25"]
    means = summary["policy_means"]
    bind("Replacing these choices", "Replacing these choices with the final checkpoints raises D's mean TEST score by "
         f"{np.mean(changes['D']):.4f} and O-S's by {np.mean(changes['O-S']):.4f}, changing their difference from "
         f"${means['P25']['O-S_minus_D']:.4f}$ to ${means['F']['O-S_minus_D']:.4f}$.")
    frequencies = [record["D_s7"]["bootstrap_selection_counts"]["9560"] / 100
                   for record in (saved["curves"], summary["VAL50_selection_diagnostics"])]
    bind("In 10{,}000 resamples", "In 10{,}000 resamples of the twenty years, D at seed 7 chooses "
         rf"update 9{{,}}560 in {frequencies[0]:.1f}\% of resamples with 25 evaluations and {frequencies[1]:.1f}\% with 50; "
         "all other resamples choose the final checkpoint.")
    for rows in (saved["curves"], summary["VAL50_selection_diagnostics"]):
        for name in ("D_s13", "D_s42"):
            require(rows[name]["bootstrap_selection_counts"][str(policies[name]["P25"])] == 10000,
                    "D earlier-choice stability")
    for clause in (
        "D at seeds 13 and 42 retains its respective earlier choice in every resample under either evaluation count.",
        "both the choices and evaluation count are fixed before any additional TEST samples are generated.",
        "Validation with 50 evaluations determines the checkpoint choices, and both the choices and evaluation count are fixed before any additional TEST samples are generated.",
        "This calculation leaves the roles of regularization and overfitting unresolved.",
        "Each occurrence of a sampled year contributes both validation months, and every checkpoint uses the same resampled years.",
        "The model weights and draws remain constant, so these frequencies measure variation across validation years.",
        "Variation under fresh sampling noise and performance on a new population remain undetermined.",
    ):
        bind(" ".join(clause.split()[:6]), clause)
    floats = tex.replace(r"\begin{figure}", r"\begin{table}").replace(r"\end{figure}", r"\end{table}")
    block = g.float_block(floats, "fig:checkpoints")
    require(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block)
            == ["figures/checkpoints/fig_checkpoints.pdf"], "checkpoint label and image differ")
    caption = ("Scores across all five checkpoints for D and O-S at the three training seeds. "
        "The subplots compare original VAL ensembles with 25 DDIM evaluations, new VAL ensembles with 50, and TEST ensembles with 50. "
        "Each point averages windows within months and then gives months equal weight. Colors distinguish methods, with open markers for D and filled markers for O-S; line styles and marker shapes distinguish seeds. "
        "Large black rings mark P25, the original validation choice, and crosses mark P50, the choice with 50 evaluations. These choices coincide for all six fits. "
        "F marks the common final optimizer update. Each ensemble has 39 draws. VAL and TEST scores concern different sets of windows. "
        "The additional TEST analysis evaluates choices already made on VAL.")
    require(g.prose_text(g.float_caption(floats, "fig:checkpoints"), source=True) == g.prose_text(caption, source=True),
            "checkpoint caption differs")
    bind("Scores across all five checkpoints", caption)
