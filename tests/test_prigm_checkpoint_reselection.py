"""Check the actual weighting, tie rule and saved-score reproduction on CPU."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/analyze_prigm_checkpoint_reselection.py"
spec = importlib.util.spec_from_file_location("checkpoint_reselection", SOURCE)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_unequal_month_windows_and_repeated_years_match_explicit_window_weights():
    months = np.array([200511, 200511, 200512, 200611, 200612, 200612, 200612])
    scores = np.array([[2., 6., 20., 10., 1., 4., 7.]])
    yearly = study.year_scores(scores, months, [2005, 2006])
    np.testing.assert_allclose(yearly, [[12., 7.]])
    # Two occurrences of 2005 and one of 2006; each month occurrence gets1/6.
    expanded = [2., 6., 20., 2., 6., 20., 10., 1., 4., 7.]
    weights = [1/12, 1/12, 1/6, 1/12, 1/12, 1/6, 1/6, 1/18, 1/18, 1/18]
    assert np.isclose(yearly[:, [0, 0, 1]].mean(), sum(v*w for v, w in zip(expanded, weights)))
    assert not np.isclose(yearly[:, [0, 0, 1]].mean(), np.mean(expanded))


def test_exact_tie_prefers_earlier_update_but_near_tie_still_uses_lower_score():
    scores = np.array([[3., 1., 2., 1., 1.], [3., 1., 2., 1., 1.-1e-13]])
    assert study.checkpoint_indices(scores).tolist() == [1, 4]


def test_saved_evidence_reproduces_private_independent_check_and_public_result():
    input_path = ROOT / study.INPUT
    before = hashlib.sha256(input_path.read_bytes()).hexdigest()
    result = study.build_result(input_path)
    # These endpoints were independently reproduced using explicit month/window weights.
    np.testing.assert_allclose(result["fixed_selection"]["percentile_2_5_97_5"],
                               [-0.03459878912304012, 0.024196675122462243], rtol=0, atol=2e-14)
    np.testing.assert_allclose(result["reselected"]["percentile_2_5_97_5"],
                               [-0.03358456092846413, 0.024587705854803566], rtol=0, atol=2e-14)
    assert result["selection_counts"]["D_s7"]["9560"] == 7168
    assert result["selection_counts"]["O-S_s13"]["22944"] == 2397
    assert result == json.loads((ROOT / study.OUTPUT).read_text())
    assert before == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert result["input"]["sha256"] == before
    assert result["source"]["sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_preserves_existing_output_before_computation(tmp_path, monkeypatch, symlink):
    output = tmp_path / "existing.json"
    if symlink:
        output.symlink_to(tmp_path / "missing.json")
    else:
        output.write_bytes(b"prior results\n")
    monkeypatch.setattr("sys.argv", [str(SOURCE), "--output", str(output)])
    def unexpected_computation(_):
        pytest.fail("an existing output must be refused before computation")
    monkeypatch.setattr(study, "build_result", unexpected_computation)
    with pytest.raises(SystemExit) as exc:
        study.main()
    assert exc.value.code == 2
    if symlink:
        assert output.is_symlink() and not output.exists()
    else:
        assert output.read_bytes() == b"prior results\n"


def test_cli_writes_a_new_result_and_refuses_a_file_created_during_computation(tmp_path, monkeypatch):
    output = tmp_path / "new.json"
    monkeypatch.setattr("sys.argv", [str(SOURCE), "--output", str(output)])
    monkeypatch.setattr(study, "build_result", lambda _: {"value": 1})
    study.main()
    assert json.loads(output.read_text()) == {"value": 1}
    output.unlink()
    def concurrent_writer(_):
        output.write_bytes(b"other result\n")
        return {"value": 2}
    monkeypatch.setattr(study, "build_result", concurrent_writer)
    with pytest.raises(FileExistsError):
        study.main()
    assert output.read_bytes() == b"other result\n"
