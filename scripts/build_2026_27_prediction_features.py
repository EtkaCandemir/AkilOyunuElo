from __future__ import annotations

"""Bridge 2026/27 fixtures to the frozen Structural ML feature schema.

The served 1X2 layer is `50% Current ML + 50% AO Domestic Poisson`. The ML
half needs rolling European form - residual windows, last-five goal rates,
venue splits - that reach back across seasons. The 2026/27 preproduction
package only carries its own season, so feeding it alone produces truncated
windows and the service silently falls back to Current AO for every row.

This script joins three sources into one leakage-safe baseline:

1. `2018/19-2025/26` European matches from the current model evaluation,
2. completed `2026/27` qualifiers from the preproduction replay,
3. the upcoming fixtures whose features are wanted.

Two properties make the output safe to serve:

- Rolling history is keyed on the permanent `club_id`, so a club carries its
  own form across seasons and across local team-id changes.
- `build_pre_match_feature_store` defers history updates to the end of each
  exact-UTC kickoff batch, so this script builds one batch at a time. An
  upcoming fixture can therefore never contribute its placeholder result to
  another upcoming fixture's rolling window.

The script changes no production parameter. It writes features, and with
`--predict` the served prediction log the production service produces from
them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from collections.abc import Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ao_elo.dynamic import (  # noqa: E402
    DynamicEloConfig,
    expected_1x2_probabilities,
    expected_score,
)
from ao_elo.ml_features import (  # noqa: E402
    build_pre_match_feature_store,
    validate_feature_store,
)
from ao_elo.production_prediction import ProductionPredictionService  # noqa: E402


PREPRODUCTION = ROOT / "data" / "season_2026_27_preproduction"
REPLAY = ROOT / "output" / "season_2026_27_preproduction"
DEFAULT_HISTORY = (
    ROOT / "output" / "current_model_evaluation_2018_2026" / "model_predictions.csv"
)
DEFAULT_IDENTITY = ROOT / "data" / "club_identity" / "team_season_identity.csv"
DEFAULT_DOMESTIC = (
    ROOT / "data" / "domestic_league_matches_2013_2026" / "domestic_matches.csv"
)
DEFAULT_FIXTURES = PREPRODUCTION / "fixtures_upcoming.csv"
DEFAULT_OUTPUT = ROOT / "output" / "season_2026_27_prediction_features"
PRODUCTION_CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"

BASELINE_COLUMNS = (
    "match_id", "season", "kickoff_utc", "competition", "stage", "tie_id",
    "is_single_match_tie", "home_team_id", "away_team_id", "home_club_id",
    "away_club_id", "home_goals", "away_goals", "actual_class",
    "actual_home_score", "is_neutral", "home_live_pre", "away_live_pre",
    "expected_home_score", "home_probability", "draw_probability",
    "away_probability", "power_delta",
)
METADATA_COLUMNS = ("match_id", "round", "round_sequence", "leg_number", "is_knockout")
SYNTHETIC_TEAM_ID_BASE = 900_000


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Structural ML feature rows for upcoming 2026/27 "
            "fixtures by bridging them to the full European match history"
        )
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--domestic-matches", type=Path, default=DEFAULT_DOMESTIC)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--match-id",
        action="append",
        default=None,
        help="restrict to one or more fixtures; repeat the flag for several",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="also run the production service and write the served log",
    )
    parser.add_argument("--generated-at-utc", type=str, default=None)
    arguments = parser.parse_args()

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    completed = load_completed_matches(arguments.history, arguments.identity)
    metadata = load_metadata(arguments.fixtures)
    fixtures = load_upcoming_fixtures(arguments.fixtures, arguments.match_id)
    domestic = load_domestic_matches(arguments.domestic_matches)

    missing_metadata = sorted(
        set(fixtures["match_id"].astype(str))
        - set(metadata["match_id"].astype(str))
    )
    if missing_metadata:
        raise ValueError(
            "Fixtures without round/leg metadata cannot be served: "
            f"{missing_metadata[:5]}"
        )

    features, batches = build_features(completed, metadata, fixtures, domestic)
    gates = validation_gates(features, fixtures, completed, batches)
    if not gates["passed"].all():
        failed = gates.loc[~gates["passed"], "gate"].tolist()
        raise ValueError(f"Feature bridge validation failed: {failed}")

    features.to_csv(output_root / "prediction_features.csv", index=False)
    gates.to_csv(output_root / "validation_gates.csv", index=False)

    served = None
    if arguments.predict:
        generated_at = (
            pd.Timestamp(arguments.generated_at_utc)
            if arguments.generated_at_utc
            else pd.Timestamp(features["kickoff_utc"].min()) - pd.Timedelta(hours=6)
        )
        service = ProductionPredictionService.from_contract(
            PRODUCTION_CONTRACT, allow_degraded_fallback=False
        )
        served = service.predict(features, generated_at_utc=generated_at)
        served.to_csv(output_root / "served_predictions.csv", index=False)

    manifest = build_manifest(features, fixtures, completed, batches, served)
    (output_root / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote prediction features to {output_root}")
    print(gates.to_string(index=False))
    if served is not None:
        print()
        status = served["prediction_status"].value_counts()
        print(status.to_string())


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def load_completed_matches(history_path: Path, identity_path: Path) -> pd.DataFrame:
    """Return every completed European match with permanent club identity."""

    historical = _historical_baseline(history_path, identity_path)
    current = _preproduction_baseline()
    combined = pd.concat([historical, current], ignore_index=True)
    combined["kickoff_utc"] = pd.to_datetime(
        combined["kickoff_utc"], utc=True, format="ISO8601"
    )
    if combined["match_id"].duplicated().any():
        raise ValueError("Completed match_id values must be unique")
    return combined.sort_values(["kickoff_utc", "match_id"], kind="stable")


def _historical_baseline(history_path: Path, identity_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(history_path)
    frame = frame.loc[frame["model"].eq("CURRENT_PRODUCTION")].copy()
    identity = pd.read_csv(identity_path)[["season", "local_team_id", "club_id"]]
    identity["local_team_id"] = identity["local_team_id"].astype(str)
    for side in ("home", "away"):
        frame[f"{side}_team_id"] = frame[f"{side}_team_id"].astype(str)
        frame = frame.merge(
            identity.rename(
                columns={"local_team_id": f"{side}_team_id", "club_id": f"{side}_club_id"}
            ),
            on=["season", f"{side}_team_id"],
            how="left",
            validate="many_to_one",
        )
    missing = int(frame[["home_club_id", "away_club_id"]].isna().sum().sum())
    if missing:
        raise ValueError(f"Historical matches missing {missing} club identities")
    frame["actual_home_score"] = _home_score(frame)
    return frame[list(BASELINE_COLUMNS)]


def _preproduction_baseline() -> pd.DataFrame:
    predictions = pd.read_csv(REPLAY / "q3_completed_replay" / "replay_predictions.csv")
    updates = pd.read_csv(REPLAY / "q3_completed_replay" / "match_updates.csv")
    source = pd.read_csv(PREPRODUCTION / "matches_completed.csv")
    clubs = _club_lookup()

    frame = predictions.merge(
        updates[
            [
                "match_id", "home_goals", "away_goals", "stage", "tie_id",
                "is_single_match_tie", "power_delta",
            ]
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    ).merge(
        source[["match_id", "is_neutral"]],
        on="match_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    frame = frame.rename(
        columns={
            "home_win_probability": "home_probability",
            "away_win_probability": "away_probability",
        }
    )
    for side in ("home", "away"):
        frame[f"{side}_club_id"] = frame[f"{side}_team_id"].map(clubs)
    missing = int(frame[["home_club_id", "away_club_id"]].isna().sum().sum())
    if missing:
        raise ValueError(f"2026/27 matches missing {missing} club identities")
    frame["is_neutral"] = frame["is_neutral"].fillna(False).astype(bool)
    frame["actual_home_score"] = _home_score(frame)
    frame["actual_class"] = np.where(
        frame["home_goals"] > frame["away_goals"], 0,
        np.where(frame["home_goals"] == frame["away_goals"], 1, 2),
    )
    return frame[list(BASELINE_COLUMNS)]


def _home_score(frame: pd.DataFrame) -> np.ndarray:
    return np.where(
        frame["home_goals"] > frame["away_goals"], 1.0,
        np.where(frame["home_goals"] == frame["away_goals"], 0.5, 0.0),
    )


def _club_lookup() -> dict[str, str]:
    audit = pd.read_csv(PREPRODUCTION / "team_identity_audit.csv")
    return dict(zip(audit["team_id"].astype(str), audit["club_id"].astype(str)))


def load_metadata(fixtures_path: Path = DEFAULT_FIXTURES) -> pd.DataFrame:
    """Round/leg metadata for completed matches and the fixtures being served.

    `fixtures_path` must be the same file the baseline rows come from.  While
    this read was pinned to DEFAULT_FIXTURES a `--fixtures` override produced
    baselines from the override and metadata from the default file, so an
    overridden fixture silently became `round=UNKNOWN, leg_number=0` and was
    scored as `LEAGUE_OR_GROUP` instead of `TWO_LEG`.
    """

    history = pd.read_csv(ROOT / "data" / "dynamic_backtest_2018_2026" / "matches.csv")
    historical = history[list(METADATA_COLUMNS)]
    sequence_by_round = _round_sequence_lookup(history)
    current = pd.read_csv(PREPRODUCTION / "matches_completed.csv")
    upcoming = pd.read_csv(fixtures_path)
    rows = []
    for frame in (current, upcoming):
        block = frame[["match_id", "round", "leg_number", "is_knockout"]].copy()
        block["round_sequence"] = _lookup_round_sequence(frame, sequence_by_round)
        rows.append(block[list(METADATA_COLUMNS)])
    return pd.concat([historical, *rows], ignore_index=True).drop_duplicates("match_id")


def _round_sequence_lookup(history: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(competition, round) -> round_sequence from the most recent full season.

    `round_sequence` is a within-season ordinal assigned over the rounds that
    season actually contains, so it cannot be computed for a season still being
    played.  The served fixtures previously took a fixed `-1.0`, a value absent
    from every training row; the model then extrapolated outside the range it
    was fitted on for every prediction it has ever served.  The competition
    format is unchanged from the latest completed season, so its ordinals are
    the defensible values to carry forward.
    """

    latest = str(history["season"].astype(str).max())
    recent = history[history["season"].astype(str).eq(latest)]
    grouped = recent.groupby(["competition", "round"])["round_sequence"]
    ambiguous = sorted(key for key, count in grouped.nunique().items() if count > 1)
    if ambiguous:
        raise ValueError(f"{latest} round_sequence is ambiguous for: {ambiguous}")
    return {
        (str(competition), str(round_name)): float(value)
        for (competition, round_name), value in grouped.min().items()
    }


def _lookup_round_sequence(
    frame: pd.DataFrame, sequence_by_round: Mapping[tuple[str, str], float]
) -> list[float]:
    """Refuse an unmapped round rather than serving the old `-1.0` sentinel."""

    missing = sorted(
        {
            (str(competition), str(round_name))
            for competition, round_name in zip(frame["competition"], frame["round"])
            if (str(competition), str(round_name)) not in sequence_by_round
        }
    )
    if missing:
        raise ValueError(
            "No round_sequence for these competition/round pairs; the round name "
            f"must match the training vocabulary: {missing}"
        )
    return [
        sequence_by_round[(str(competition), str(round_name))]
        for competition, round_name in zip(frame["competition"], frame["round"])
    ]


def load_upcoming_fixtures(
    path: Path, match_ids: list[str] | None
) -> pd.DataFrame:
    fixtures = pd.read_csv(path)
    if match_ids:
        fixtures = fixtures.loc[fixtures["match_id"].isin(match_ids)]
        unknown = sorted(set(match_ids) - set(fixtures["match_id"]))
        if unknown:
            raise ValueError(f"Unknown fixture match_id: {unknown}")
    if fixtures.empty:
        raise ValueError("No upcoming fixtures selected")
    ratings = pd.read_csv(REPLAY / "playoff_pre_match_team_ratings.csv")
    live = dict(zip(ratings["team_id"], ratings["pre_playoff_live_elo"]))
    clubs = _club_lookup()
    config = DynamicEloConfig.calibrated_v2()

    rows = []
    for fixture in fixtures.itertuples(index=False):
        for side in ("home", "away"):
            if getattr(fixture, f"{side}_team_id") not in live:
                raise ValueError(
                    f"No pre-playoff rating for {getattr(fixture, f'{side}_team_id')}"
                )
        home = float(live[fixture.home_team_id])
        away = float(live[fixture.away_team_id])
        neutral = bool(fixture.is_neutral)
        single = bool(fixture.is_single_match_tie)
        expected = expected_score(home, away, config, neutral=neutral)
        probabilities = expected_1x2_probabilities(
            home, away, config, neutral=neutral, is_single_match_tie=single
        )
        rows.append(
            {
                "match_id": fixture.match_id,
                "season": fixture.season,
                "kickoff_utc": fixture.kickoff_utc,
                "competition": fixture.competition,
                "stage": fixture.stage,
                "tie_id": fixture.tie_id,
                "is_single_match_tie": single,
                "home_team_id": fixture.home_team_id,
                "away_team_id": fixture.away_team_id,
                "home_club_id": clubs[str(fixture.home_team_id)],
                "away_club_id": clubs[str(fixture.away_team_id)],
                # Placeholders. Each fixture is built in its own kickoff batch,
                # so these never reach another emitted row's rolling window.
                "home_goals": 0,
                "away_goals": 0,
                "actual_class": 1,
                "actual_home_score": 0.5,
                "power_delta": 0.0,
                "is_neutral": neutral,
                "home_live_pre": home,
                "away_live_pre": away,
                "expected_home_score": expected,
                "home_probability": probabilities[0],
                "draw_probability": probabilities[1],
                "away_probability": probabilities[2],
            }
        )
    frame = pd.DataFrame(rows)[list(BASELINE_COLUMNS)]
    frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, format="ISO8601")
    return frame.sort_values(["kickoff_utc", "match_id"], kind="stable")


def load_domestic_matches(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    return frame.dropna(subset=["home_ao_club_id", "away_ao_club_id"])


# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------


def build_features(
    completed: pd.DataFrame,
    metadata: pd.DataFrame,
    fixtures: pd.DataFrame,
    domestic: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Build one feature row per fixture, one exact-UTC batch at a time."""

    blocks = []
    batches = 0
    for kickoff, batch in fixtures.groupby("kickoff_utc", sort=True):
        baseline = pd.concat([completed, batch], ignore_index=True)
        baseline = _integer_team_ids(baseline)
        baseline = baseline.sort_values(["kickoff_utc", "match_id"], kind="stable")
        store = build_pre_match_feature_store(
            baseline.reset_index(drop=True), domestic, match_metadata=metadata
        )
        validate_feature_store(store, expected_rows=len(baseline))
        blocks.append(store.loc[store["match_id"].isin(batch["match_id"])])
        batches += 1
    features = pd.concat(blocks, ignore_index=True)
    return features.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True), batches


def _integer_team_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Map team ids to integers for the feature store's join column.

    Rolling history is keyed on `club_id`, so the integer is a carrier only.
    The 2026/27 package uses `AO-UEFA-*` identifiers, which the feature store
    cannot cast, and a stable synthetic id keeps the two eras in one frame.
    """

    result = frame.copy()
    synthetic: dict[str, int] = {}

    def as_integer(value: object) -> int:
        text = str(value)
        if text.isdigit():
            return int(text)
        if text not in synthetic:
            synthetic[text] = SYNTHETIC_TEAM_ID_BASE + len(synthetic)
        return synthetic[text]

    for side in ("home", "away"):
        result[f"{side}_team_id"] = result[f"{side}_team_id"].map(as_integer)
    return result


# ---------------------------------------------------------------------------
# validation and reporting
# ---------------------------------------------------------------------------


def validation_gates(
    features: pd.DataFrame,
    fixtures: pd.DataFrame,
    completed: pd.DataFrame,
    batches: int,
) -> pd.DataFrame:
    rows = [
        {
            "gate": "one_row_per_fixture",
            "passed": bool(
                len(features) == len(fixtures)
                and set(features["match_id"]) == set(fixtures["match_id"])
            ),
            "observed": len(features),
            "requirement": "Exactly one feature row per requested fixture",
        },
        {
            "gate": "one_batch_per_kickoff",
            "passed": bool(batches == fixtures["kickoff_utc"].nunique()),
            "observed": batches,
            "requirement": "Fixtures never share a build with a later kickoff",
        },
        {
            "gate": "history_reaches_back",
            # A debutant club legitimately has no European history, so the
            # gate asks that the bridge is working in general, not that every
            # single club has a past.
            "passed": bool(
                features["home_euro_matches_pre"].median() >= 10.0
                and features["away_euro_matches_pre"].median() >= 10.0
            ),
            "observed": float(
                min(
                    features["home_euro_matches_pre"].median(),
                    features["away_euro_matches_pre"].median(),
                )
            ),
            "requirement": "Median club carries at least ten prior European matches",
        },
        {
            "gate": "completed_history_present",
            "passed": bool(len(completed) > 6000),
            "observed": len(completed),
            "requirement": "Bridge includes the multi-season match store",
        },
        {
            # Reported, not enforced. A club with no prior match anywhere has
            # no rest or form value to give. The frozen pipeline imputes those
            # inputs, so the row still serves the full ensemble - it does not
            # fall back - which means `fallback_rate` alone will never reveal
            # it. Operators need this count separately.
            "gate": "rows_with_imputed_model_input",
            "passed": True,
            "observed": int(len(_non_finite_model_features(features))),
            "requirement": "Reported only: imputed inputs still serve the ensemble",
        },
    ]
    return pd.DataFrame(rows)


def _model_feature_names() -> list[str]:
    """Ask the frozen artifact which columns it actually consumes."""

    from ao_elo.production_prediction import (
        _feature_schema_payload,
        load_production_prediction_runtime,
    )

    runtime = load_production_prediction_runtime(PRODUCTION_CONTRACT)
    schema = _feature_schema_payload(runtime.ml_model)
    return list(schema.get("numeric", []))


def _non_finite_model_features(features: pd.DataFrame) -> pd.DataFrame:
    names = [name for name in _model_feature_names() if name in features.columns]
    values = features[names].apply(pd.to_numeric, errors="coerce")
    mask = ~np.isfinite(values.to_numpy(float))
    return features.loc[mask.any(axis=1), ["match_id"]]


def build_manifest(
    features: pd.DataFrame,
    fixtures: pd.DataFrame,
    completed: pd.DataFrame,
    batches: int,
    served: pd.DataFrame | None,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "layer": "STRUCTURAL_ML_FEATURE_BRIDGE_2026_27",
        "changes_production_parameters": False,
        "rating_feedback": False,
        "fixtures": int(len(fixtures)),
        "kickoff_batches": int(batches),
        "completed_matches_in_history": int(len(completed)),
        "history_window": [
            str(completed["season"].min()),
            str(completed["season"].max()),
        ],
        "median_home_euro_matches_pre": float(
            features["home_euro_matches_pre"].median()
        ),
        "median_away_euro_matches_pre": float(
            features["away_euro_matches_pre"].median()
        ),
    }
    if served is not None:
        manifest["served_status_counts"] = {
            str(key): int(value)
            for key, value in served["prediction_status"].value_counts().items()
        }
        manifest["rows_with_imputed_model_input"] = int(
            len(_non_finite_model_features(features))
        )
        manifest["served_coverage_counts"] = {
            str(key): int(value)
            for key, value in served["domestic_poisson_coverage"].value_counts().items()
        }
    return manifest


if __name__ == "__main__":
    main()
