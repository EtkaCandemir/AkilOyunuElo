from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ao_elo.dynamic import initialize_season, run_season
from ao_elo.dynamic_csv import (
    MATCH_INPUT_COLUMNS,
    MATCH_UPDATE_COLUMNS,
    RATINGS_STATE_COLUMNS,
    REPLAY_PREDICTION_COLUMNS,
    STATE_CHECKPOINT_FILENAME,
    load_selected_v2_config,
    read_matches,
    read_team_seeds,
    run_batch,
    state_from_frame,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "contracts" / "ao_european_elo_v2_production.json"
)


def write_inputs(root: Path) -> tuple[Path, Path]:
    ratings_path = root / "initial_ratings.csv"
    matches_path = root / "matches.csv"
    pd.DataFrame(
        [
            {
                "season": "2026/27",
                "team_id": "A",
                "team_name": "Alpha",
                "ao_first_elo": 1500.0,
            },
            {
                "season": "2026/27",
                "team_id": "B",
                "team_name": "Beta",
                "ao_first_elo": 1400.0,
            },
        ]
    ).to_csv(ratings_path, index=False)
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "season": "2026/27",
                "kickoff_utc": "2026-09-01T19:00:00Z",
                "competition": "UCL",
                "round": "League Stage",
                "tie_id": "",
                "is_knockout": False,
                "is_tie_decider": False,
                "stage": "LEAGUE",
                "home_team_id": "A",
                "away_team_id": "B",
                "home_goals": 2,
                "away_goals": 1,
                "xg_home": 2.4,
                "xg_away": 0.7,
                "xg_analysis_eligible": True,
                "is_neutral": False,
                "decided_on_penalties": False,
                "advanced_team_id": "",
            },
            {
                "match_id": "m2",
                "season": "2026/27",
                "kickoff_utc": "2026-09-08T19:00:00Z",
                "competition": "UCL",
                "round": "League Stage",
                "tie_id": "",
                "is_knockout": False,
                "is_tie_decider": False,
                "stage": "LEAGUE",
                "home_team_id": "B",
                "away_team_id": "A",
                "home_goals": 1,
                "away_goals": 1,
                "xg_home": "",
                "xg_away": "",
                "xg_analysis_eligible": False,
                "is_neutral": False,
                "decided_on_penalties": False,
                "advanced_team_id": "",
            },
        ],
        columns=MATCH_INPUT_COLUMNS,
    ).to_csv(matches_path, index=False)
    return ratings_path, matches_path


def test_manifest_loads_final_selected_parameters() -> None:
    config = load_selected_v2_config(MANIFEST)

    assert config.elo_scale == pytest.approx(835.5614973262034)
    assert config.home_advantage == pytest.approx(148.54426619132505)
    assert config.k_factor == pytest.approx(103.98098633392752)
    assert config.power_carry == 0.0
    assert config.draw_at_even == pytest.approx(0.24)
    assert config.draw_shape == pytest.approx(1.0)
    assert config.goal_difference_enabled is True
    assert config.goal_alpha == pytest.approx(0.15)
    assert config.goal_tau == pytest.approx(300.0)
    assert config.goal_difference_cap == 4
    assert config.xg_performance_enabled is True
    assert config.xg_performance_ratio == pytest.approx(0.30)
    assert config.xg_performance_scale == pytest.approx(1.25)
    assert config.minimum_winner_gain_ratio == pytest.approx(0.70)
    assert config.progression_bonus_enabled is True
    assert config.progression_base_bonus == pytest.approx(12.0)
    assert config.progression_stages_per_competition == 4
    assert config.fixed_progression_config.increment("UEL") == pytest.approx(8.0)
    assert config.fixed_progression_config.cap("UCL") == pytest.approx(48.0)
    assert config.achievement_reserve is None


def test_inactive_optional_layer_cannot_hide_nonzero_parameter(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["goal_margin"]["active"] = False
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Inactive goal-difference"):
        load_selected_v2_config(manifest)


def test_production_manifest_requires_active_1x2_output(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["one_x_two_probability"]["active"] = False
    manifest = tmp_path / "inactive-probability.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="activate the calibrated 1X2"):
        load_selected_v2_config(manifest)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("competition_k", "active"),
    ],
)
def test_production_manifest_rejects_unapproved_dynamic_layers(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload[section][field] = True
    manifest = tmp_path / "unapproved-layer.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must remain disabled"):
        load_selected_v2_config(manifest)


def test_production_manifest_rejects_invalid_progression_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["progression_bonus"]["loser_deduction"] = True
    manifest = tmp_path / "invalid-progression.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot deduct"):
        load_selected_v2_config(manifest)


def test_production_manifest_rejects_knockout_playoff_bonus_stage(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["progression_bonus"]["eligible_stages"].insert(
        0, "KNOCKOUT_PLAYOFF"
    )
    payload["progression_bonus"]["stages_per_competition"] = 5
    manifest = tmp_path / "invalid-kpo-progression.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="KNOCKOUT_PLAYOFF is not eligible"):
        load_selected_v2_config(manifest)


def test_production_manifest_rejects_invalid_xg_contract(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["xg_performance"]["minimum_winner_gain_ratio"] = 0.60
    manifest = tmp_path / "invalid-xg.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="1-max_xg_ratio"):
        load_selected_v2_config(manifest)


def test_production_manifest_rejects_prediction_rating_feedback(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["prediction_layer"]["rating_feedback"] = True
    manifest = tmp_path / "invalid-prediction-layer.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot change AO Live Elo"):
        load_selected_v2_config(manifest)


def test_production_manifest_rejects_achievement_reserve_reactivation(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["achievement_reserve"] = {"active": True}
    manifest = tmp_path / "reserve-reactivated.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="experimental-only"):
        load_selected_v2_config(manifest)


def test_manifest_requires_frozen_version_and_real_json_booleans(
    tmp_path: Path,
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["model_version"] = "v2-candidate"
    manifest = tmp_path / "wrong-version.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_selected_v2_config(manifest)

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["goal_margin"]["active"] = "false"
    manifest = tmp_path / "wrong-boolean.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON boolean"):
        load_selected_v2_config(manifest)


def test_csv_batch_and_python_api_use_identical_kernel(tmp_path: Path) -> None:
    ratings_path, matches_path = write_inputs(tmp_path)
    output = tmp_path / "output"
    config = load_selected_v2_config(MANIFEST)

    season, seeds = read_team_seeds(ratings_path)
    api_initial = initialize_season(season, seeds, config)
    api_state, api_updates = run_season(
        api_initial,
        read_matches(matches_path),
        config,
    )
    csv_state, csv_updates = run_batch(
        ratings_path,
        matches_path,
        output,
        config,
    )

    assert csv_state == api_state
    assert csv_updates == api_updates
    assert list(pd.read_csv(output / "ratings_state.csv").columns) == (
        RATINGS_STATE_COLUMNS
    )
    assert list(pd.read_csv(output / "match_updates.csv").columns) == (
        MATCH_UPDATE_COLUMNS
    )
    assert pd.read_csv(output / "match_updates.csv")[
        "expected_score_semantics"
    ].eq("normalized match points: win=1, draw=0.5, loss=0").all()
    audit = pd.read_csv(output / "match_updates.csv")
    assert audit["goal_difference_enabled"].all()
    assert audit["goal_alpha"].eq(0.15).all()
    assert audit["goal_tau"].eq(300.0).all()
    assert audit["goal_difference_cap"].eq(4).all()
    assert audit["xg_performance_enabled"].all()
    assert audit["xg_applied"].tolist() == [True, False]
    assert audit["xg_fallback_used"].tolist() == [False, True]
    assert audit.loc[0, "xg_power_adjustment"] != pytest.approx(0.0)
    assert list(pd.read_csv(output / "replay_predictions.csv").columns) == (
        REPLAY_PREDICTION_COLUMNS
    )
    assert (output / STATE_CHECKPOINT_FILENAME).exists()
    assert not (output / "pre_match_log.csv").exists()
    manifest = json.loads((output / "batch_manifest.json").read_text())
    assert manifest["mode"] == "RETROSPECTIVE_REPLAY"
    assert manifest["prospective_holdout_evidence"] is False


def test_same_csv_input_produces_byte_identical_outputs(tmp_path: Path) -> None:
    ratings_path, matches_path = write_inputs(tmp_path)
    config = load_selected_v2_config(MANIFEST)
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_batch(ratings_path, matches_path, first, config)
    run_batch(ratings_path, matches_path, second, config)

    assert (first / "ratings_state.csv").read_bytes() == (
        second / "ratings_state.csv"
    ).read_bytes()
    assert (first / "match_updates.csv").read_bytes() == (
        second / "match_updates.csv"
    ).read_bytes()
    assert (first / "replay_predictions.csv").read_bytes() == (
        second / "replay_predictions.csv"
    ).read_bytes()
    assert (first / STATE_CHECKPOINT_FILENAME).read_bytes() == (
        second / STATE_CHECKPOINT_FILENAME
    ).read_bytes()
    assert (first / "batch_manifest.json").read_bytes() == (
        second / "batch_manifest.json"
    ).read_bytes()


def test_batch_rejects_directory_with_legacy_pre_match_log(tmp_path: Path) -> None:
    ratings_path, matches_path = write_inputs(tmp_path)
    output = tmp_path / "legacy-output"
    output.mkdir()
    (output / "pre_match_log.csv").write_text("legacy\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be mistaken"):
        run_batch(
            ratings_path,
            matches_path,
            output,
            load_selected_v2_config(MANIFEST),
        )


def test_batch_cli_writes_contract_files(tmp_path: Path) -> None:
    ratings_path, matches_path = write_inputs(tmp_path)
    output = tmp_path / "cli-output"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_dynamic_elo.py"),
            "--initial-ratings",
            str(ratings_path),
            "--matches",
            str(matches_path),
            "--output-dir",
            str(output),
            "--model-manifest",
            str(MANIFEST),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Matches processed: 2" in result.stdout
    assert (output / "ratings_state.csv").exists()
    assert (output / "match_updates.csv").exists()
    assert (output / "replay_predictions.csv").exists()
    assert (output / STATE_CHECKPOINT_FILENAME).exists()
    assert not (output / "pre_match_log.csv").exists()


def test_invalid_boolean_and_duplicate_match_are_rejected(tmp_path: Path) -> None:
    _, matches_path = write_inputs(tmp_path)
    data = pd.read_csv(matches_path)
    data["is_neutral"] = data["is_neutral"].astype(object)
    data.loc[0, "is_neutral"] = "yes"
    data.to_csv(matches_path, index=False)

    with pytest.raises(ValueError, match="is_neutral"):
        read_matches(matches_path)

    _, matches_path = write_inputs(tmp_path)
    data = pd.read_csv(matches_path)
    data.loc[1, "match_id"] = "m1"
    data.to_csv(matches_path, index=False)
    with pytest.raises(ValueError, match="duplicate match_id"):
        read_matches(matches_path)


def test_xg_inputs_require_an_eligible_two_team_pair(tmp_path: Path) -> None:
    _, matches_path = write_inputs(tmp_path)
    data = pd.read_csv(matches_path)
    data.loc[0, "xg_away"] = float("nan")
    data.to_csv(matches_path, index=False)

    with pytest.raises(ValueError, match="provided together"):
        read_matches(matches_path)

    _, matches_path = write_inputs(tmp_path)
    data = pd.read_csv(matches_path)
    data.loc[0, "xg_analysis_eligible"] = False
    data.to_csv(matches_path, index=False)
    with pytest.raises(ValueError, match="Ineligible xG"):
        read_matches(matches_path)


def test_saved_state_rejects_config_mismatch(tmp_path: Path) -> None:
    ratings_path, matches_path = write_inputs(tmp_path)
    output = tmp_path / "output"
    config = load_selected_v2_config(MANIFEST)
    run_batch(ratings_path, matches_path, output, config)
    frame = pd.read_csv(output / "ratings_state.csv")
    frame.loc[:, "config_id"] = "wrong"

    with pytest.raises(ValueError, match="does not match"):
        state_from_frame(frame, config)


# ---------------------------------------------------------------------------
# knockout fixtures must state their tie format: a missing flag used to default
# to False, which silently priced every single-match tie - finals included -
# with the two-leg draw intercept
# ---------------------------------------------------------------------------


PILOT_MATCHES = ROOT / "data" / "pilot_20_teams" / "matches.csv"


def write_without_format_flag(frame: pd.DataFrame, path: Path) -> Path:
    frame.drop(columns=["is_single_match_tie"]).to_csv(path, index=False)
    return path


def test_knockout_row_without_the_format_flag_is_rejected(tmp_path: Path) -> None:
    source = pd.read_csv(PILOT_MATCHES)
    target = write_without_format_flag(source, tmp_path / "matches.csv")

    with pytest.raises(ValueError, match="is_single_match_tie is required"):
        read_matches(target)


def test_blank_format_flag_is_rejected_like_a_missing_column(
    tmp_path: Path,
) -> None:
    """An empty cell is just as unknown as an absent column."""
    source = pd.read_csv(PILOT_MATCHES)
    source["is_single_match_tie"] = source["is_single_match_tie"].astype(object)
    source.loc[source["is_knockout"], "is_single_match_tie"] = ""
    target = tmp_path / "matches.csv"
    source.to_csv(target, index=False)

    with pytest.raises(ValueError, match="is_single_match_tie is required"):
        read_matches(target)


def test_league_rows_do_not_need_the_format_flag(tmp_path: Path) -> None:
    """Only knockout ties have a format, so league rows stay unaffected."""
    source = pd.read_csv(PILOT_MATCHES)
    league_only = source.loc[~source["is_knockout"].astype(bool)]
    target = write_without_format_flag(league_only, tmp_path / "matches.csv")

    matches = read_matches(target)

    assert len(matches) == len(league_only)
    assert not any(match.is_single_match_tie for match in matches)


def test_pilot_finals_are_marked_as_single_match_ties() -> None:
    matches = read_matches(PILOT_MATCHES)
    single = [match for match in matches if match.is_single_match_tie]

    assert single, "the pilot must exercise the single-match draw intercept"
    assert all(match.round == "Final" for match in single)


def test_single_match_finals_use_the_shorter_draw_intercept() -> None:
    """The whole point of the flag: finals must not be priced as two-leg ties."""
    from ao_elo.dynamic import effective_draw_at_even

    config = load_selected_v2_config(MANIFEST)
    matches = read_matches(PILOT_MATCHES)
    finals = [match for match in matches if match.is_single_match_tie]

    for match in finals:
        assert effective_draw_at_even(config, match.is_single_match_tie) == pytest.approx(
            config.single_match_draw_at_even
        )
    assert config.single_match_draw_at_even < config.draw_at_even
