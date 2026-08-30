"""2026/27 AO First Elo tablosunu okunabilir bir PDF olarak uretir.

Sayilar CSV'den ve aktif config'ten okunur, belgede sabit yazilmaz.
"""

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
    bullets,
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

DATA = ROOT / "output" / "season_2026_27_preproduction" / "ao_first_elo_2026_27.csv"

SPEC = PdfSpec(
    filename="AO_2026_27_AO_First_Elo_Tam_Dokuman.pdf",
    title="AO First Elo 2026/27",
    subtitle="Sezon Başı Ratingleri ve Bileşen Dökümü",
    version="AO European Elo v2 | Sezon başı ratingleri",
    document_date="30 Ağustos 2026",
    subject="2026/27 sezonu AO First Elo değerleri ve her bileşenin dökümü",
)


def _n(value: object, digits: int = 1) -> str:
    """Turkce ondalik ayraci; tabloyu okuyan kisi virgul bekler."""
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", " ").replace(".", ",")


def main() -> None:
    output_path, docs_path = build_pdf(SPEC, story())
    print(f"PDF written: {output_path}")
    print(f"PDF synced:  {docs_path}")


def story() -> list[object]:
    s = styles()
    cfg = AOEuropeanEloConfig.active()
    data = pd.read_csv(DATA).sort_values("ao_first_elo_rank")

    out: list[object] = cover(
        SPEC,
        [
            ["Sezon", str(data["season"].iloc[0])],
            ["Model sürümü", str(data["model_version"].iloc[0])],
            ["Takım sayısı", str(len(data))],
            [
                "Elo aralığı",
                f"{_n(data['ao_first_elo'].min())} – {_n(data['ao_first_elo'].max())}",
            ],
            [
                "Kupa dağılımı",
                " / ".join(
                    f"{k} {v}" for k, v in data["competition"].value_counts().items()
                ),
            ],
            ["Teknik otorite", "contracts/ao_european_elo_v2_production.json"],
        ],
        s,
        summary_title="Bu belge ne gösteriyor",
        summary=(
            "AO First Elo, sezon başındaki kulüp gücüdür — hiçbir 2026/27 maçı "
            "oynanmadan önceki tahmin. Belge iki tablo içerir: sıralama ve her "
            "ratingin hangi bileşenlerden kurulduğunun dökümü. Maçlar başladıktan "
            "sonra bu değerler AO Live Elo'ya evrilir; bu belge o evrimin "
            "başlangıç noktasıdır."
        ),
    )

    out += _explanation(s, cfg, data)
    out += _worked_example(s, data)
    out += _ranking(s, data)
    out += _components(s, data)
    out += _summaries(s, data)
    return out


def _explanation(s: dict, cfg, data: pd.DataFrame) -> list[object]:
    return [
        h1("1. Rating nasıl kuruluyor", s),
        body(
            "Her kulüp için iki ayrı tahmin üretilir. Domestic Prior, kulübün kendi "
            "ülkesindeki kanıttan gelir: ligin Avrupa'daki gücü, kulübün geçen "
            "sezonki lig sırası ve kupa başarısı. European Prior ise kulübün "
            "Avrupa'daki geçmiş performansından gelir.",
            s,
        ),
        body(
            "İkisi, kulübün Avrupa'ya ne kadar maruz kaldığına göre karıştırılır. "
            "Hiç Avrupa oynamamış bir kulüp tamamen yerel tahminle başlar; çok "
            "oynamış bir kulüpte Avrupa kanıtı ağır basar ama tamamen devralmaz.",
            s,
        ),
        formula(
            [
                "AO First Elo = Domestic Prior",
                "             + effective_exposure * (European Prior - Domestic Prior)",
                "             + Domestic Surprise duzeltmesi",
                "",
                f"effective_exposure = min(european_exposure, "
                f"{str(cfg.max_european_exposure).replace('.', ',')})",
            ],
            s,
        ),
        callout(
            "Avrupa geçmişi neden bir tavanda toplanmıyor",
            f"european_tail_beta = {str(cfg.european_tail_beta).replace('.', ',')} "
            "olduğu için Avrupa geçmişi normu kesilmez. Önceki `0` değeri, "
            f"benchmark'ı ({str(cfg.european_history_benchmark).replace('.0', '')}) "
            "aşan bütün kulüpleri tek bir European Prior'a indiriyordu — güçlü ve "
            "çok güçlü Avrupa sicilleri aynı sayıya düşüyordu. Şimdi ayrışık "
            "kalıyorlar.",
            s,
            tone="green",
        ),
        callout(
            f"Yerel kanıt neden en az %{int((1 - cfg.max_european_exposure) * 100)} ağırlıkta kalıyor",
            "Kulüp ne kadar çok Avrupa oynamış olursa olsun Domestic Prior'ın ağırlığı "
            f"{str(1 - cfg.max_european_exposure).replace('.', ',')}'in altına inmez. "
            "Sebep istatistiksel değil yapısaldır: bir kulübün Avrupa maç sayısı, "
            "gücünün yanı sıra turnuvada ne kadar ilerlediğinin de fonksiyonudur. O "
            "iki şeyi birbirinden ayırmak için elde yeterli kanıt yok, bu yüzden "
            "yerel kanıt tamamen devre dışı bırakılmıyor. Bu tabloda "
            f"{int((data['effective_european_exposure'] >= cfg.max_european_exposure - 1e-9).sum())} "
            "kulüp bu tavana dayanmış durumda.",
            s,
            tone="blue",
        ),
        h2("1.1 Tablolardaki sütunlar", s),
        table(
            [
                ["Sütun", "Anlamı"],
                [
                    "Lig sırası",
                    "Geçen sezon kendi liginde bitirdiği sıra. Boşsa kulüp güncel üst "
                    "ligde bulunamamıştır (genelde kupa yoluyla girmiş 2. lig kulübü); "
                    "tahmini sıra yazılmaz.",
                ],
                [
                    "Başarı",
                    "Domestic Achievement: lig sırası ve kupa başarısının birleşimi. "
                    f"0 ile {str(cfg.achievement_cap).replace('.', ',')} arasında.",
                ],
                [
                    "Yerel tahmin",
                    "Domestic Prior — yalnız ülke/lig/yerel başarıdan gelen rating.",
                ],
                [
                    "Avrupa tahmini",
                    "European Prior — yalnız Avrupa geçmişinden gelen rating.",
                ],
                [
                    "Maruziyet",
                    "Avrupa tahminine verilen ağırlık. 0 ise kulüp saf yerel "
                    f"projeksiyondur, {str(cfg.max_european_exposure).replace('.', ',')} tavandır.",
                ],
                [
                    "Sürpriz",
                    "Domestic Surprise: kulüp kendi geçmiş sıralamasına göre beklenenden "
                    f"iyi/kötü bitirdiyse uygulanan düzeltme, ±{int(cfg.domestic_surprise_max_abs_adjustment)} ile sınırlı.",
                ],
                ["AO First Elo", "Nihai sezon başı rating."],
            ],
            [3.9 * cm, 12.55 * cm],
            s,
        ),
        page_break(),
    ]


def _worked_example(s: dict, data: pd.DataFrame) -> list[object]:
    """Tek bir kulubu ucdan uca gostermek, tabloyu okunabilir kilan sey."""
    row = data.iloc[0]
    exposure = float(row["effective_european_exposure"])
    # Domestic Surprise blend'den ONCE Domestic Prior'a girer, sonra degil:
    # AO First = (dp + D) + e * (ep - dp - D). AO First uzerindeki net etkisi
    # bu yuzden D degil, D * (1 - e) olur.
    without_surprise = float(row["domestic_prior"]) + exposure * (
        float(row["european_prior"]) - float(row["domestic_prior"])
    )
    blended = float(row["adjusted_domestic_prior"]) + exposure * (
        float(row["european_prior"]) - float(row["adjusted_domestic_prior"])
    )
    return [
        h1("2. Bir kulüp uçtan uca", s),
        body(
            f"Tablodaki sayıların nasıl kurulduğunu görmek için sıralamanın başındaki "
            f"kulübü baştan sona izleyelim: {row['team_name']}.",
            s,
        ),
        table(
            [
                ["Adım", "Değer"],
                ["Lig sırası", f"{_n(row['domestic_position'], 0)} / {_n(row['league_team_count'], 0)}"],
                ["Lig yüzdeliği", _n(row["domestic_position_percentile"], 4)],
                ["Lig bitiriş skoru", _n(row["league_finish_score"], 4)],
                ["Kupa katkısı", _n(row["cup_double_bonus"], 4)],
                ["Domestic Achievement", _n(row["domestic_achievement_score"], 4)],
                ["Lig gücü", _n(row["league_strength"], 4)],
                ["→ Domestic Prior", _n(row["domestic_prior"], 2)],
                ["Domestic Surprise düzeltmesi", _n(row["domestic_surprise_domestic_adjustment"], 2)],
                ["→ Düzeltilmiş Domestic Prior", _n(row["adjusted_domestic_prior"], 2)],
                ["Avrupa geçmişi (ağırlıklı)", _n(row["weighted_european_history"], 2)],
                ["Katılım normalize oranı", _n(row["european_history_rate"], 4)],
                ["→ European Prior", _n(row["european_prior"], 2)],
                ["Sezon maruziyeti", _n(row["weighted_season_exposure"], 4)],
                ["Maç maruziyeti", _n(row["weighted_match_exposure"], 4)],
                ["→ Efektif maruziyet", _n(exposure, 4)],
                ["→ AO First Elo", _n(row["ao_first_elo"], 2)],
            ],
            [8.2 * cm, 8.25 * cm],
            s,
        ),
        body(
            f"Sıra önemli: Domestic Surprise düzeltmesi karışımdan önce Domestic "
            f"Prior'a girer, sonra değil. Kulübün yerel tahmini "
            f"{_n(row['domestic_prior'])} idi; sürpriz düzeltmesiyle "
            f"{_n(row['adjusted_domestic_prior'])} oldu. Avrupa tahmini "
            f"{_n(row['european_prior'])} ve aradaki farkın "
            f"%{_n(exposure * 100, 0)}'i alınarak nihai rating "
            f"{_n(row['ao_first_elo'])} çıkıyor.",
            s,
        ),
        callout(
            "Sürprizin nihai rating üzerindeki etkisi düzeltmenin kendisi kadar değildir",
            f"Domestic Prior'a eklenen {_n(row['domestic_surprise_domestic_adjustment'], 2)} "
            f"puanın bir kısmı karışımda Avrupa tahminine yer bırakırken erir. "
            f"Sürpriz olmasaydı rating {_n(without_surprise)} olurdu; gerçek değer "
            f"{_n(row['ao_first_elo'])}. Yani net etki "
            f"{_n(row['domestic_surprise_ao_first_elo_adjustment'], 2)} puandır — "
            f"düzeltmenin (1 − maruziyet) katı. Tablodaki \"Sürpriz\" sütunu bu net "
            "etkiyi gösterir.",
            s,
            tone="green",
        ),
        page_break(),
    ]


def _ranking(s: dict, data: pd.DataFrame) -> list[object]:
    out: list[object] = [h1("3. Sıralama", s)]
    rows = [["Sıra", "Takım", "Ülke", "Kupa", "AO First Elo"]]
    for r in data.itertuples(index=False):
        rows.append(
            [
                str(int(r.ao_first_elo_rank)),
                str(r.team_name),
                str(r.country_code),
                str(r.competition),
                _n(r.ao_first_elo),
            ]
        )
    out.append(table(rows, [1.7 * cm, 7.3 * cm, 1.9 * cm, 2.2 * cm, 3.35 * cm], s))
    out.append(page_break())
    return out


def _components(s: dict, data: pd.DataFrame) -> list[object]:
    out: list[object] = [
        h1("4. Bileşen dökümü", s),
        body(
            "Aynı sıralama, her ratingin hangi parçalardan kurulduğuyla birlikte. "
            "Sütun anlamları için bölüm 1.1'e bakın.",
            s,
        ),
    ]
    header = [
        "Sıra",
        "Takım",
        "Lig\nsırası",
        "Başarı",
        "Yerel\ntahmin",
        "Avrupa\ntahmini",
        "Maruz.",
        "Sürpriz",
        "AO First",
    ]
    rows = [header]
    for r in data.itertuples(index=False):
        position = (
            "—"
            if pd.isna(r.domestic_position)
            else f"{int(r.domestic_position)}/{int(r.league_team_count)}"
        )
        rows.append(
            [
                str(int(r.ao_first_elo_rank)),
                str(r.team_name),
                position,
                _n(r.domestic_achievement_score, 2),
                _n(r.domestic_prior, 0),
                _n(r.european_prior, 0),
                _n(r.effective_european_exposure, 2),
                _n(r.domestic_surprise_ao_first_elo_adjustment, 1),
                _n(r.ao_first_elo, 1),
            ]
        )
    widths = [1.2 * cm, 4.35 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 1.8 * cm, 1.4 * cm, 1.5 * cm, 1.5 * cm]
    out.append(table(rows, widths, s))
    out.append(page_break())
    return out


def _summaries(s: dict, data: pd.DataFrame) -> list[object]:
    by_country = (
        data.groupby("country")
        .agg(takim=("team_id", "count"), ortalama=("ao_first_elo", "mean"), en_yuksek=("ao_first_elo", "max"))
        .sort_values("ortalama", ascending=False)
        .head(20)
    )
    country_rows = [["Ülke", "Takım", "Ortalama", "En yüksek"]]
    for name, r in by_country.iterrows():
        country_rows.append([str(name), str(int(r.takim)), _n(r.ortalama), _n(r.en_yuksek)])

    source_rows = [["Kanıt türü", "Takım", "Anlamı"]]
    meanings = {
        "European Evidence-Based Rating": "Avrupa geçmişi baskın",
        "Mixed Domestic-European Estimate": "Yerel ve Avrupa kanıtı karışık",
        "Pure Domestic Projection": "Hiç Avrupa kanıtı yok",
    }
    for name, count in data["rating_source_type"].value_counts().items():
        source_rows.append([str(name), str(int(count)), meanings.get(str(name), "")])

    surprise_rows = [["Domestic Surprise", "Takım", "Anlamı"]]
    surprise_meanings = {
        "APPLIED": "Düzeltme uygulandı",
        "INSUFFICIENT_HISTORY": "Beş sezondan az geçmiş; düzeltme yok",
        "NO_CHANGE": "Geçmişiyle uyumlu; düzeltme sıfır",
    }
    for name, count in data["domestic_surprise_status"].value_counts().items():
        surprise_rows.append([str(name), str(int(count)), surprise_meanings.get(str(name), "")])

    return [
        h1("5. Dağılımlar", s),
        h2("5.1 Ülkeye göre (ortalamaya göre ilk 20)", s),
        table(country_rows, [6.2 * cm, 2.6 * cm, 3.8 * cm, 3.85 * cm], s),
        Spacer(1, 0.35 * cm),
        h2("5.2 Ratingin dayandığı kanıt", s),
        table(source_rows, [7.0 * cm, 2.2 * cm, 7.25 * cm], s),
        Spacer(1, 0.35 * cm),
        h2("5.3 Domestic Surprise durumu", s),
        table(surprise_rows, [5.4 * cm, 2.2 * cm, 8.85 * cm], s),
        Spacer(1, 0.4 * cm),
        callout(
            "Bu belgenin sınırı",
            "AO First Elo bir sezon başı tahminidir, sonuç değildir. 2026/27 maçları "
            "oynandıkça kulüplerin gücü AO Live Elo ile güncellenir ve bu tablodan "
            "ayrışır. Buradaki sıralama, hiçbir maç oynanmadan önce modelin ne "
            "düşündüğünü gösterir.",
            s,
            tone="amber",
        ),
    ]


if __name__ == "__main__":
    main()
