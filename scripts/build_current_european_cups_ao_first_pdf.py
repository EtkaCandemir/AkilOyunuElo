"""Build the current 2026/27 UCL/UEL/UECL roster CSV and AO First PDF."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import Spacer

from pdf_common import (
    PdfSpec,
    body,
    build_pdf,
    callout,
    cover,
    formula,
    h1,
    h2,
    page_break,
    styles,
    table,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ao_elo.config import AOEuropeanEloConfig  # noqa: E402


AO_FIRST = ROOT / "output" / "season_2026_27_preproduction" / "ao_first_elo_2026_27.csv"
PREPRODUCTION = ROOT / "data" / "season_2026_27_preproduction"
OUTPUT_ROOT = ROOT / "output" / "season_2026_27_current_european_cups"
OUTPUT_CSV = OUTPUT_ROOT / "ao_first_elo_current_european_teams_2026_27.csv"

SPEC = PdfSpec(
    filename="AO_2026_27_Guncel_Avrupa_Kupalari_AO_First_Elo.pdf",
    title="2026/27 Güncel Avrupa Kupaları",
    subtitle="UCL, UEL ve UECL Lig Aşaması — AO First Elo",
    version="AO European Elo v2 | Güncel Avrupa kupaları",
    document_date="30 Ağustos 2026",
    subject="2026/27 UCL, UEL ve UECL lig aşamasındaki 108 kulübün AO First Elo dökümü",
)


def _n(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", " ").replace(".", ",")


def _competition_path(events: pd.DataFrame) -> pd.Series:
    return events.groupby("team_id")["competition"].agg(
        lambda values: ">".join(dict.fromkeys(values))
    )


def build_current_roster() -> pd.DataFrame:
    ao = pd.read_csv(AO_FIRST)
    completed = pd.read_csv(PREPRODUCTION / "matches_completed.csv")
    fixtures = pd.read_csv(PREPRODUCTION / "fixtures_upcoming.csv")

    fixture_sides = pd.concat(
        [
            fixtures[
                ["competition", "kickoff_utc", "match_id", "home_team_id"]
            ].rename(columns={"home_team_id": "team_id"}),
            fixtures[
                ["competition", "kickoff_utc", "match_id", "away_team_id"]
            ].rename(columns={"away_team_id": "team_id"}),
        ],
        ignore_index=True,
    )
    current = (
        fixture_sides.groupby(["competition", "team_id"], as_index=False)
        .agg(
            league_stage_fixture_count=("match_id", "nunique"),
            first_league_stage_kickoff_utc=("kickoff_utc", "min"),
            last_league_stage_kickoff_utc=("kickoff_utc", "max"),
        )
        .rename(columns={"competition": "current_competition"})
    )
    if current["team_id"].duplicated().any():
        raise ValueError("A club appears in more than one current competition")

    completed_sides = pd.concat(
        [
            completed[["kickoff_utc", "competition", "home_team_id"]].rename(
                columns={"home_team_id": "team_id"}
            ),
            completed[["kickoff_utc", "competition", "away_team_id"]].rename(
                columns={"away_team_id": "team_id"}
            ),
        ],
        ignore_index=True,
    ).sort_values(["kickoff_utc", "team_id"], kind="stable")
    paths = _competition_path(completed_sides).rename("played_competition_path")

    ao = ao.rename(columns={"competition": "entry_competition"})
    data = current.merge(ao, on="team_id", how="left", validate="one_to_one")
    data = data.join(paths, on="team_id")
    if data["team_name"].isna().any():
        missing = data.loc[data["team_name"].isna(), "team_id"].tolist()
        raise ValueError(f"Current clubs missing AO First rows: {missing}")

    def qualification_status(row: pd.Series) -> str:
        if row["entry_competition"] != row["current_competition"]:
            return f"TRANSFERRED_FROM_{row['entry_competition']}"
        if str(row["entry_round"]).endswith("-LS"):
            return "DIRECT_LEAGUE_STAGE"
        return "QUALIFIED_SAME_COMPETITION"

    data["qualification_status"] = data.apply(qualification_status, axis=1)
    data["current_roster_rank"] = (
        data["ao_first_elo"].rank(method="first", ascending=False).astype(int)
    )
    data["current_competition_rank"] = (
        data.groupby("current_competition")["ao_first_elo"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    data["roster_source"] = "UEFA_2026_27_LEAGUE_STAGE_FIXTURES"

    counts = data["current_competition"].value_counts().to_dict()
    if counts != {"UCL": 36, "UEL": 36, "UECL": 36}:
        raise ValueError(f"Current competition counts are not 36/36/36: {counts}")
    expected_fixtures = {"UCL": 8, "UEL": 8, "UECL": 6}
    for competition, expected in expected_fixtures.items():
        observed = data.loc[
            data["current_competition"].eq(competition), "league_stage_fixture_count"
        ]
        if not observed.eq(expected).all():
            raise ValueError(
                f"{competition} clubs do not all have {expected} league-stage fixtures"
            )

    leading = [
        "season",
        "model_version",
        "current_roster_rank",
        "current_competition",
        "current_competition_rank",
        "team_id",
        "team_name",
        "country",
        "country_code",
        "entry_competition",
        "entry_round",
        "qualification_status",
        "played_competition_path",
        "league_stage_fixture_count",
        "first_league_stage_kickoff_utc",
        "last_league_stage_kickoff_utc",
        "roster_source",
    ]
    remainder = [column for column in data.columns if column not in leading]
    return data[leading + remainder].sort_values("current_roster_rank", kind="stable")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = build_current_roster()
    data.to_csv(OUTPUT_CSV, index=False, lineterminator="\n")
    output_path, docs_path = build_pdf(SPEC, story(data))
    print(f"CSV written: {OUTPUT_CSV}")
    print(f"PDF written: {output_path}")
    print(f"PDF synced:  {docs_path}")


def story(data: pd.DataFrame) -> list[object]:
    s = styles()
    cfg = AOEuropeanEloConfig.active()
    fixture_counts = {
        competition: int(group["league_stage_fixture_count"].sum() // 2)
        for competition, group in data.groupby("current_competition")
    }
    out: list[object] = cover(
        SPEC,
        [
            ["Sezon", str(data["season"].iloc[0])],
            ["Model sürümü", str(data["model_version"].iloc[0])],
            ["Devam eden takım", str(len(data))],
            ["Güncel kupa dağılımı", "UCL 36 / UEL 36 / UECL 36"],
            [
                "Lig aşaması fikstürü",
                " / ".join(f"{key} {fixture_counts[key]}" for key in ("UCL", "UEL", "UECL")),
            ],
            [
                "AO First Elo aralığı",
                f"{_n(data['ao_first_elo'].min())} – {_n(data['ao_first_elo'].max())}",
            ],
            ["Roster kaynağı", "2026/27 UEFA lig aşaması fikstürleri"],
        ],
        s,
        summary_title="Bu belge ne gösteriyor",
        summary=(
            "Bu belge, 30 Ağustos 2026 itibarıyla UCL, UEL veya UECL lig "
            "aşamasında devam eden 108 kulübü gösterir. Takım seçimi günceldir; "
            "rating sütunu ise sezon başlamadan önce donmuş AO First Elo'dur. "
            "Current competition ile entry competition ayrı tutulur: elemelerde "
            "kupa değiştiren kulüpler yanlış kupada gösterilmez."
        ),
    )
    out += _scope(s, data)
    out += _worked_example(s, data)
    out += _ranking(s, data)
    out += _components(s, data)
    out += _summaries(s, cfg, data)
    return out


def _scope(s: dict, data: pd.DataFrame) -> list[object]:
    status = data["qualification_status"].value_counts()
    return [
        h1("1. Evren ve kolonların anlamı", s),
        body(
            "Ana AO First dosyası sezon başında Avrupa'ya giren 237 kulübü içerir. "
            "Bu belge o dosyayı güncel lig aşaması fikstürleriyle kesiştirir; bu "
            "nedenle yalnız hâlâ Avrupa kupalarında devam eden 108 takım kalır.",
            s,
        ),
        table(
            [
                ["Kolon", "Anlam"],
                ["current_competition", "Kulübün güncel lig aşaması: UCL, UEL veya UECL"],
                ["entry_competition", "Kulübün sezona başladığı kupa"],
                ["entry_round", "Sezona girdiği tur; örneğin CL-LS veya CL-Q1"],
                ["qualification_status", "Doğrudan, elemeden veya başka kupadan transfer"],
                ["AO First Elo", "Hiçbir 2026/27 maçı oynanmadan önceki rating"],
            ],
            [4.6 * cm, 11.85 * cm],
            s,
        ),
        h2("1.1 Lig aşamasına geliş yolları", s),
        table(
            [
                ["Yol", "Takım"],
                ["Doğrudan lig aşaması", str(int(status.get("DIRECT_LEAGUE_STAGE", 0)))],
                [
                    "Aynı kupada elemeleri geçti",
                    str(int(status.get("QUALIFIED_SAME_COMPETITION", 0))),
                ],
                [
                    "UCL'den alt kupaya geçti",
                    str(int(status.get("TRANSFERRED_FROM_UCL", 0))),
                ],
                [
                    "UEL'den UECL'ye geçti",
                    str(int(status.get("TRANSFERRED_FROM_UEL", 0))),
                ],
            ],
            [10.0 * cm, 6.45 * cm],
            s,
        ),
        callout(
            "AO First ile AO Live aynı şey değildir",
            "Buradaki Elo değerleri kulüplerin sezon başı başlangıç gücüdür. Temmuz "
            "ve Ağustos elemeleri AO Live Elo'yu değiştirmiştir; bu belge güncel "
            "takım evrenini sezon başı AO First ölçeğinde karşılaştırır.",
            s,
            tone="amber",
        ),
        page_break(),
    ]


def _worked_example(s: dict, data: pd.DataFrame) -> list[object]:
    row = data.iloc[0]
    exposure = float(row["effective_european_exposure"])
    return [
        h1("2. En yüksek ratingli kulüp", s),
        body(
            f"Güncel 108 takım içinde sezon başı AO First Elo'su en yüksek kulüp "
            f"{row['team_name']}. Aşağıdaki tablo ratingin ana bileşenlerini ve "
            "kulübün güncel turnuva durumunu birlikte gösterir.",
            s,
        ),
        table(
            [
                ["Alan", "Değer"],
                ["Güncel kupa", str(row["current_competition"])],
                ["Giriş kupası / turu", f"{row['entry_competition']} / {row['entry_round']}"],
                ["Lig aşamasına geliş", str(row["qualification_status"])],
                ["Lig sırası", f"{_n(row['domestic_position'], 0)} / {_n(row['league_team_count'], 0)}"],
                ["Domestic Achievement", _n(row["domestic_achievement_score"], 4)],
                ["Domestic Prior", _n(row["domestic_prior"], 2)],
                ["Düzeltilmiş Domestic Prior", _n(row["adjusted_domestic_prior"], 2)],
                ["European Prior", _n(row["european_prior"], 2)],
                ["Efektif Avrupa maruziyeti", _n(exposure, 4)],
                ["Domestic Surprise net etkisi", _n(row["domestic_surprise_ao_first_elo_adjustment"], 2)],
                ["AO First Elo", _n(row["ao_first_elo"], 2)],
            ],
            [8.2 * cm, 8.25 * cm],
            s,
        ),
        formula(
            [
                "AO First Elo = Adjusted Domestic Prior",
                "             + effective_exposure",
                "               * (European Prior - Adjusted Domestic Prior)",
            ],
            s,
        ),
        page_break(),
    ]


def _ranking(s: dict, data: pd.DataFrame) -> list[object]:
    rows = [["108", "237", "Takım", "Ülke", "Güncel", "Giriş", "Yol", "AO First"]]
    labels = {
        "DIRECT_LEAGUE_STAGE": "Doğrudan",
        "QUALIFIED_SAME_COMPETITION": "Elemeden",
        "TRANSFERRED_FROM_UCL": "UCL transfer",
        "TRANSFERRED_FROM_UEL": "UEL transfer",
    }
    for row in data.itertuples(index=False):
        rows.append(
            [
                str(int(row.current_roster_rank)),
                str(int(row.ao_first_elo_rank)),
                str(row.team_name),
                str(row.country_code),
                str(row.current_competition),
                str(row.entry_competition),
                labels[str(row.qualification_status)],
                _n(row.ao_first_elo),
            ]
        )
    return [
        h1("3. Güncel 108 takımın AO First sıralaması", s),
        body(
            "108 sütunu yalnız devam eden kulüpler arasındaki sıradır. 237 sütunu "
            "sezon başındaki tam katılımcı evrenindeki özgün AO First sırasıdır.",
            s,
        ),
        table(
            rows,
            [0.8 * cm, 0.8 * cm, 4.1 * cm, 1.15 * cm, 1.35 * cm, 1.25 * cm, 3.0 * cm, 2.0 * cm],
            s,
        ),
        page_break(),
    ]


def _components(s: dict, data: pd.DataFrame) -> list[object]:
    rows = [["Sıra", "Takım", "Kupa", "Lig", "Başarı", "Yerel", "Avrupa", "Maruz.", "Sürpriz", "AO First"]]
    for row in data.itertuples(index=False):
        position = (
            "—"
            if pd.isna(row.domestic_position)
            else f"{int(row.domestic_position)}/{int(row.league_team_count)}"
        )
        rows.append(
            [
                str(int(row.current_roster_rank)),
                str(row.team_name),
                str(row.current_competition),
                position,
                _n(row.domestic_achievement_score, 2),
                _n(row.domestic_prior, 0),
                _n(row.european_prior, 0),
                _n(row.effective_european_exposure, 2),
                _n(row.domestic_surprise_ao_first_elo_adjustment, 1),
                _n(row.ao_first_elo, 1),
            ]
        )
    return [
        h1("4. Bileşen dökümü", s),
        body(
            "Her kulübün AO First Elo'sunu oluşturan ana parçalar. Güncel kupa "
            "kolonu lig aşaması roster'ından gelir; rating bileşenleri sezon başında "
            "donmuştur.",
            s,
        ),
        table(
            rows,
            [0.8 * cm, 3.6 * cm, 1.1 * cm, 1.15 * cm, 1.3 * cm, 1.35 * cm, 1.45 * cm, 1.2 * cm, 1.3 * cm, 1.45 * cm],
            s,
        ),
        page_break(),
    ]


def _summaries(s: dict, cfg: AOEuropeanEloConfig, data: pd.DataFrame) -> list[object]:
    summary = (
        data.groupby("current_competition")
        .agg(
            takim=("team_id", "count"),
            ortalama=("ao_first_elo", "mean"),
            medyan=("ao_first_elo", "median"),
            en_yuksek=("ao_first_elo", "max"),
            en_dusuk=("ao_first_elo", "min"),
        )
        .reindex(["UCL", "UEL", "UECL"])
    )
    summary_rows = [["Kupa", "Takım", "Ortalama", "Medyan", "En yüksek", "En düşük"]]
    for competition, row in summary.iterrows():
        summary_rows.append(
            [
                competition,
                str(int(row.takim)),
                _n(row.ortalama),
                _n(row.medyan),
                _n(row.en_yuksek),
                _n(row.en_dusuk),
            ]
        )

    route = pd.crosstab(data["current_competition"], data["qualification_status"])
    route_rows = [["Güncel kupa", "Doğrudan", "Aynı kupada eleme", "UCL transfer", "UEL transfer"]]
    for competition in ("UCL", "UEL", "UECL"):
        route_rows.append(
            [
                competition,
                str(int(route.get("DIRECT_LEAGUE_STAGE", pd.Series()).get(competition, 0))),
                str(int(route.get("QUALIFIED_SAME_COMPETITION", pd.Series()).get(competition, 0))),
                str(int(route.get("TRANSFERRED_FROM_UCL", pd.Series()).get(competition, 0))),
                str(int(route.get("TRANSFERRED_FROM_UEL", pd.Series()).get(competition, 0))),
            ]
        )

    top_rows = [["Kupa", "Sıra", "Takım", "AO First"]]
    for competition in ("UCL", "UEL", "UECL"):
        subset = data.loc[data["current_competition"].eq(competition)].nsmallest(
            10, "current_competition_rank"
        )
        for row in subset.itertuples(index=False):
            top_rows.append(
                [
                    competition,
                    str(int(row.current_competition_rank)),
                    str(row.team_name),
                    _n(row.ao_first_elo),
                ]
            )

    return [
        h1("5. Kupa özetleri", s),
        h2("5.1 AO First dağılımı", s),
        table(summary_rows, [2.2 * cm, 2.0 * cm, 3.0 * cm, 3.0 * cm, 3.1 * cm, 3.15 * cm], s),
        Spacer(1, 0.35 * cm),
        h2("5.2 Lig aşamasına geliş yolu", s),
        table(route_rows, [3.0 * cm, 3.0 * cm, 4.0 * cm, 3.2 * cm, 3.25 * cm], s),
        page_break(),
        h1("6. Her kupada ilk 10", s),
        table(top_rows, [2.0 * cm, 1.7 * cm, 8.5 * cm, 4.25 * cm], s),
        Spacer(1, 0.4 * cm),
        callout(
            "Belgenin sınırı",
            "Roster 30 Ağustos 2026 itibarıyla günceldir; UEFA sayfaları sportif "
            "listeyi lisans ve disiplin kararları bakımından provisional olarak "
            "tanımlar. AO First Elo ise sezon başında donmuş başlangıç ratingidir. "
            f"Avrupa exposure tavanı {_n(cfg.max_european_exposure, 2)} dahil hiçbir "
            "model parametresi bu belge için yeniden hesaplanmamıştır.",
            s,
            tone="amber",
        ),
    ]


if __name__ == "__main__":
    main()
