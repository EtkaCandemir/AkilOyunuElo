from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo import AOEuropeanEloConfig, compute_ao_first_elo_from_csv  # noqa: E402
from ao_elo.config import V2_RATING_MULTIPLIER  # noqa: E402
from ao_elo.dynamic_csv import load_selected_v2_config, run_batch  # noqa: E402


OUTPUT_ROOT = ROOT / "output" / "v2_pilots"
DYNAMIC_OUTPUT = OUTPUT_ROOT / "dynamic_replay"
MANIFEST = (
    ROOT / "contracts" / "ao_european_elo_v2_production.json"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    outputs: dict[str, pd.DataFrame] = {}
    for name, data_dir in (
        ("synthetic", ROOT / "data" / "pilot_10_teams"),
        ("real", ROOT / "data" / "real_pilot_10_teams"),
    ):
        v1 = compute(data_dir, AOEuropeanEloConfig.v1_1())
        v2 = compute(data_dir, AOEuropeanEloConfig.active())
        expected = 500.0 + V2_RATING_MULTIPLIER * (v1["ao_first_elo"] - 500.0)
        np.testing.assert_allclose(v2["ao_first_elo"], expected, atol=1e-9, rtol=0)
        assert v2["ao_first_elo_rank"].tolist() == v1["ao_first_elo_rank"].tolist()
        assert not v2["ao_first_elo"].isin([500.0, 2000.0]).any()
        np.testing.assert_allclose(
            v2.loc[v2["european_exposure"].eq(0), "ao_first_elo"],
            v2.loc[v2["european_exposure"].eq(0), "domestic_prior"],
            atol=1e-12,
            rtol=0,
        )
        path = OUTPUT_ROOT / f"{name}_ao_first_elo_v2.csv"
        v2.to_csv(path, index=False)
        outputs[name] = v2
        summaries.append(
            {
                "pilot": name,
                "teams": len(v2),
                "minimum": v2["ao_first_elo"].min(),
                "median": v2["ao_first_elo"].median(),
                "maximum": v2["ao_first_elo"].max(),
                "validation_warning_rows": int(
                    v2["validation_warnings"].fillna("").ne("").sum()
                ),
                "v1_v2_rank_identical": True,
            }
        )

    initial_path = OUTPUT_ROOT / "dynamic_initial_ratings.csv"
    outputs["real"][["season", "team_id", "team_name", "ao_first_elo"]].to_csv(
        initial_path,
        index=False,
    )
    dynamic_config = load_selected_v2_config(MANIFEST)
    final_state, updates = run_batch(
        initial_path,
        ROOT / "data" / "dynamic_pilot" / "matches.csv",
        DYNAMIC_OUTPUT,
        dynamic_config,
    )
    assert len(updates) == 5
    assert dynamic_config.power_carry == 0.0
    assert all(update.goal_multiplier == 1.0 for update in updates)
    assert all(update.progression_reserve_added == 0.0 for update in updates)
    assert all(update.trophy_reserve_added == 0.0 for update in updates)
    assert all(
        abs(
            update.home_win_probability
            + update.draw_probability
            + update.away_win_probability
            - 1.0
        )
        <= 1e-12
        for update in updates
    )
    assert all(
        abs(
            update.home_win_probability
            + 0.5 * update.draw_probability
            - update.expected_home_score
        )
        <= 1e-12
        for update in updates
    )
    summaries_frame = pd.DataFrame(summaries)
    summaries_frame.to_csv(OUTPUT_ROOT / "pilot_summary.csv", index=False)
    write_report(summaries_frame, outputs["real"], final_state, updates)
    print(summaries_frame.to_string(index=False))
    print(f"Dynamic matches: {len(updates)}")
    print(f"Output: {OUTPUT_ROOT}")


def compute(data_dir: Path, config: AOEuropeanEloConfig) -> pd.DataFrame:
    return compute_ao_first_elo_from_csv(
        teams_csv=data_dir / "teams.csv",
        country_coefficients_csv=data_dir / "country_coefficients.csv",
        domestic_context_csv=data_dir / "domestic_context.csv",
        club_european_points_csv=data_dir / "club_european_points.csv",
        config=config,
    )


def write_report(
    summaries: pd.DataFrame,
    real: pd.DataFrame,
    final_state,
    updates,
) -> None:
    ranking = real[["ao_first_elo_rank", "team_name", "ao_first_elo"]]
    lines = [
        "# AO European Elo v2 Pilot Sonuclari",
        "",
        "Model: `ao-european-elo-v2.0-dev-freeze`",
        "",
        "V2, v1.1'in 500 etrafındaki affine ölçek dönüşümüdür; bu nedenle pilot "
        "takım sırası birebir korunur. 500-2000 bir referans bandıdır, hard cap değildir.",
        "",
        "## Statik özet",
        "",
        dataframe_to_markdown(summaries),
        "",
        "## Gerçek pilot sıralaması",
        "",
        dataframe_to_markdown(ranking),
        "",
        "## Dinamik smoke test",
        "",
        f"- İşlenen sentetik exact-UTC maç: {len(updates)}",
        "- Power güncellemeleri sıfır toplamlıdır.",
        "- Season carry kapalıdır: power_carry=0.",
        "- H/D/A olasılıkları toplamı 1 ve expected-score kimliği korunur.",
        "- Goal margin katmanı kapalı: bütün G değerleri 1.0.",
        "- Achievement Reserve kapalı: bütün reserve ekleri 0.0.",
        "- Tahmin çıktısı retrospective replay'dir; holdout kanıtı değildir.",
        f"- Final state takım sayısı: {len(final_state.ratings)}",
        "",
    ]
    (OUTPUT_ROOT / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    headers = [str(column) for column in dataframe.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in dataframe.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


if __name__ == "__main__":
    main()
