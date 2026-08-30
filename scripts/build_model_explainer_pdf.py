"""AO European Elo v2 ayrintili model anlatimini PDF olarak uretir.

Sayisal degerler contract ve aktif config'ten okunur; belgede sabit yazilmaz.
Boylece contract degistiginde PDF yeniden uretildiginde kendiliginden duzelir
ve bir sonraki okuyucu bayat bir sayiya guvenmez.
"""

from __future__ import annotations

import json
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
from ao_elo.production_prediction import (  # noqa: E402
    selected_production_prediction_config,
)

CONTRACT = ROOT / "contracts" / "ao_european_elo_v2_production.json"
DOCUMENT_DATE = "30 Ağustos 2026"

SPEC = PdfSpec(
    filename="AkilOyunu_Elo_v2_Ayrintili_Model_Anlatimi.pdf",
    title="AO European Elo v2",
    subtitle="Ayrıntılı Model Anlatımı",
    version="AO European Elo v2 | Ayrıntılı model anlatımı",
    document_date=DOCUMENT_DATE,
    subject="AO European Elo v2 rating ve tahmin katmanlarının uçtan uca teknik anlatımı",
)


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _find(node: object, key: str) -> object:
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                return value
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _num(value: object) -> str:
    return f"{float(value):g}"


def _effective_prediction_weights(c: dict) -> tuple[float, float, float]:
    layer = c["prediction_layer"]
    top = layer["top_level_blend"]
    ml_component = layer["current_ml_component"]
    poisson_component = layer["ao_domestic_poisson_component"]
    effective_ao = top["current_ml_weight"] * ml_component["ao_weight"] + (
        top["ao_domestic_poisson_weight"] * poisson_component["ao_weight"]
    )
    effective_ml = top["current_ml_weight"] * ml_component["ml_weight"]
    effective_poisson = (
        top["ao_domestic_poisson_weight"] * poisson_component["poisson_weight"]
    )
    return effective_ao, effective_ml, effective_poisson


def main() -> None:
    output_path, docs_path = build_pdf(SPEC, story())
    print(f"PDF written: {output_path}")
    print(f"PDF synced:  {docs_path}")


def story() -> list[object]:
    s = styles()
    c = _contract()
    cfg = AOEuropeanEloConfig.active()
    pred = selected_production_prediction_config()
    layer = c["prediction_layer"]
    top = layer["top_level_blend"]
    ml_component = layer["current_ml_component"]
    poisson_component = layer["ao_domestic_poisson_component"]
    core = c["dynamic_core"]

    eff_ao, eff_ml, eff_poisson = _effective_prediction_weights(c)

    out: list[object] = cover(
        SPEC,
        [
            ["Rating sürümü", c["model_version"]],
            ["Production revision", c["production_revision"]],
            ["Revision türü", c["runtime_input_guards"]["revision_kind"]],
            ["Tahmin sürümü", layer["model_version"]],
            ["Tahmin kararı", layer["decision"]],
            ["Aktivasyon", layer["activation_date"]],
            ["Teknik otorite", "contracts/ao_european_elo_v2_production.json"],
        ],
        s,
        summary_title="Belgenin amacı",
        summary=(
            "Bu belge modelin iki ayrı katmanını uçtan uca anlatır: kulüp gücünü "
            "üreten rating katmanı ve o gücü olasılığa çeviren tahmin katmanı. "
            "İki katman birbirine karıştırılmaz; tahmin çıktısı rating'e geri "
            "beslenmez. Bütün sayısal değerler contract ve aktif config'ten "
            "okunmuştur, eski rapor veya sunumlardan değil."
        ),
    )

    out += _section_overview(s)
    out += _section_first_elo(s, cfg, c)
    out += _section_power(s, core, c)
    out += _section_margin_xg(s, c)
    out += _section_draw(s, c)
    out += _section_live(s, c)
    out += _section_prediction(s, c, pred, eff_ao, eff_ml, eff_poisson)
    out += _section_monitoring(s)
    out += _section_layers(s, c, cfg, (eff_ao, eff_ml, eff_poisson))
    out += _section_evaluation(s)
    out += _section_limits(s)
    out += _section_holdout(s, cfg, core, top, ml_component, poisson_component)
    out += _section_operations(s)
    return out


def _section_overview(s: dict) -> list[object]:
    return [
        h1("1. Model ne yapar", s),
        body(
            "AO European Elo iki ayrı soruyu cevaplar. Birincisi \"bu kulüp şu anda "
            "ne kadar güçlü?\", ikincisi \"bu maçın sonucu ne olur?\". Sistem bu iki "
            "soruyu ayrı katmanlarda çözer ve ayrım belgenin tamamında korunur.",
            s,
        ),
        body(
            "Sezon başında her kulübün bir başlangıç gücü olmalıdır, ama Avrupa'da "
            "yeni oynayan bir kulübün Avrupa geçmişi yoktur. AO First Elo bu boşluğu "
            "doldurur: kulübün kendi ülkesindeki lig gücü ve yerel başarısı bir "
            "yerel tahmin üretir, Avrupa geçmişi ayrı bir tahmin üretir, ve ikisi "
            "kulübün Avrupa'ya ne kadar maruz kaldığına göre karıştırılır.",
            s,
        ),
        body(
            "Sezon başladıktan sonra Power Elo devreye girer: her maçta iki takım "
            "arasında sıfır toplamlı bir puan alışverişi olur. Beklenmedik sonuçlar "
            "daha çok puan hareket ettirir. Gol farkı bu hareketi sınırlı biçimde "
            "büyütür; xG varsa küçük bir performans düzeltmesi eklenir.",
            s,
        ),
        h2("Uçtan uca akış", s),
        formula(
            [
                "ham sezon verisi",
                "  -> kimlik cozumlemesi (permanent club_id)",
                "  -> domestic / european statik ozellikler",
                "  -> AO First Elo                (sezon basi guc)",
                "  -> Power Elo mac replay'i      (mac mac guncelleme)",
                "  -> Achievement Reserve + Progression Bonus",
                "  -> AO Live Elo                 (servis edilen guc)",
                "  -> Current AO 1X2              (ratingten olasilik)",
                "  -> Structural ML  ve  Domestic Poisson",
                "  -> servis edilen ensemble",
                "  -> prediction log",
            ],
            s,
        ),
        callout(
            "Aynı kickoff batch'i ve sızıntı koruması",
            "Aynı anda başlayan maçlar tek bir pre-match snapshot ile işlenir. Sırayla "
            "işlenselerdi ikinci maç birincinin sonucundan güncellenmiş ratingi "
            "görürdü; oysa gerçekte ikisi aynı anda oynanıyordu. Batch içindeki tüm "
            "maçlar aynı rating durumundan okur, güncellemeler batch sonunda uygulanır.",
            s,
            tone="amber",
        ),
        page_break(),
    ]


def _section_first_elo(s: dict, cfg, c: dict) -> list[object]:
    weights = " / ".join(
        _num(cfg.season_weights[k])
        for k in ("t_minus_4", "t_minus_3", "t_minus_2", "t_minus_1", "t")
    )
    return [
        h1("2. AO First Elo — sezon başı güç", s),
        h2("2.1 Domestic Prior", s),
        body(
            "Kulübün kendi ülkesindeki kanıttan üretilen tahmin. Önce ülke gücü "
            f"normalize edilir; son beş sezonun UEFA ülke puanları {weights} "
            "ağırlıklarıyla (en eskiden en yeniye) toplanır.",
            s,
        ),
        formula(
            [
                "League Strength = (ln(1 + weighted_country_score) "
                f"/ ln(1 + {_num(cfg.country_strength_benchmark)})) ^ {_num(cfg.gamma)}",
                "",
                f"Achievement Scale = {_num(cfg.achievement_alpha)} + "
                f"{_num(1 - cfg.achievement_alpha)} * League Strength",
                "",
                "Domestic Prior = "
                f"{_num(cfg.base_rating)}",
                f"    + {cfg.domestic_league_component:.10f} * League Strength",
                f"    + {cfg.domestic_achievement_component:.10f} "
                "* Domestic Achievement * Achievement Scale",
            ],
            s,
        ),
        body(
            "Achievement Scale, aynı yerel başarının güçlü bir ligde daha fazla kanıt "
            "taşımasını sağlar: zayıf ligde şampiyon olmak, güçlü ligde şampiyon "
            "olmakla aynı şey değildir.",
            s,
        ),
        h2("2.2 Lig sırası ve kupa katkısı", s),
        formula(
            [
                "Position Percentile = (team_count - position) / (team_count - 1)",
                f"League Finish Score = {_num(cfg.percentile_floor)} + "
                f"{_num(cfg.percentile_scale)} * Position Percentile",
                "  (sampiyon ise taban " f"{_num(cfg.champion_base_score)})",
                "",
                "Domestic Achievement = min(cap, max(L, C) + w * min(L, C))",
                f"  L = League Finish Score,  C = {_num(cfg.cup_base_score)} (kupa)",
                f"  w = {cfg.cup_contribution_weight!r}",
                f"  cap = {_num(cfg.achievement_cap)}",
            ],
            s,
        ),
        body(
            "Ağırlık seçilmemiş, türetilmiştir: "
            f"w = {_num(cfg.cup_double_bonus_multiplier)} * "
            f"{_num(cfg.champion_base_score)} / {_num(cfg.cup_base_score)}. "
            "Amaç, önceki kuralın zaten ödüllendirdiği grubu birebir korumaktır — "
            "şampiyon artı kupa toplamı iki kuralda da 1,0800'dir.",
            s,
        ),
        callout(
            "max tek başına neden yetmiyordu",
            "max(L, C) kupayı bir taban yapar. Lig skoru 0,62'nin üstünde olan bir "
            "kupa şampiyonu kupasından sıfır kredi alır, ve etki ters yönlüdür: "
            "ligde ne kadar kötüysen kupa o kadar değerli olur. Katkı terimi bu "
            "boşluğu kapatır ve iki başarının zayıf olanını gerçek bir katkı olarak "
            "ekler.",
            s,
            tone="green",
        ),
        h2("2.3 Katılım normalizasyonu ve exposure", s),
        formula(
            [
                "European Prior = "
                f"{_num(cfg.base_rating)} + {cfg.european_prior_max_boost:.9f} "
                "* european_history_norm",
                "",
                "rate = history * (1 + k) / (weighted_season_exposure + k)",
                f"  k = {_num(cfg.european_participation_shrinkage)}",
                "",
                "european_exposure = "
                f"{_num(cfg.exposure_season_weight)} * season + "
                f"{_num(cfg.exposure_match_weight)} * match",
                f"effective = min(european_exposure, {_num(cfg.max_european_exposure)})",
                "",
                "AO First Elo = Domestic Prior",
                "             + effective * (European Prior - Domestic Prior)",
            ],
            s,
        ),
        body(
            "Katılım normalizasyonunun çözdüğü problem şudur: iki kulüp beş sezonda "
            "aynı toplam Avrupa puanını toplamışsa ama biri beş sezonun beşinde, "
            "diğeri ikisinde oynamışsa, ikisinin Avrupa gücü aynı değildir. Ham "
            "geçmiş toplamı bu farkı görmez.",
            s,
        ),
        body(
            "(1 + k) payı katmanın güvenliğini sağlar: tam katılımda oran yayımlanan "
            "geçmişe birebir eşit olur, yani beş sezonun beşinde oynamış bir kulüp "
            "hiç hareket etmez. Düzeltme yalnız gerçek katılım açığıyla orantılıdır.",
            s,
        ),
        callout(
            "Exposure tavanının anlamı",
            f"Kulüp ne kadar çok Avrupa oynamış olursa olsun yerel kanıtın ağırlığı "
            f"{_num(1 - cfg.max_european_exposure)}'in altına inmez. Sebep "
            "istatistiksel değil yapısaldır: Avrupa maç sayısı kulübün gücünün yanı "
            "sıra ne kadar ileri gittiğinin de fonksiyonudur, ve o iki şeyi "
            "birbirinden ayırmak için elde yeterli kanıt yoktur.",
            s,
            tone="blue",
        ),
        h2("2.4 İnvariantlar", s),
        table(
            [
                ["İnvariant", "Neden"],
                ["Tam katılımda normalizasyon nötrdür", "(1+k)/(1+k) = 1"],
                [
                    "Hiç katılmamış kulüpte oran sıfırdır",
                    "O satırlar european_exposure = 0 taşır; prior blend tarafından "
                    "zaten yok sayılır",
                ],
                [
                    "Normalizasyon rating'i düşürmez",
                    "Payda (pw + k) her zaman (1 + k) ile sınırlı",
                ],
                ["Kupa kazanmayanda katkı sıfırdır", "min(L, 0) = 0"],
                [
                    "Kupa katkısı bir taban değildir",
                    "max hâlâ tabanı verir; katkı onun üstüne eklenir",
                ],
                [
                    "league_finish_score floor altına düşmez",
                    f"Bilinen ve bilinmeyen sırada taban {_num(cfg.percentile_floor)}",
                ],
                [
                    "achievement_cap bir korkuluktur",
                    f"{_num(cfg.achievement_cap)} erişilebilir maksimum değildir; "
                    "aktif ağırlıkta ulaşılabilir en yüksek değer 1,0800",
                ],
            ],
            [6.2 * cm, 10.25 * cm],
            s,
        ),
        page_break(),
    ]


def _section_power(s: dict, core: dict, c: dict) -> list[object]:
    q = c["qualification_transition"]
    base = q["base_stage_importance_multipliers"]
    eff = q["stage_k_multipliers"]
    retention = q["qualifier_delta_retention"]
    rows = [["Tur", "Base importance", "Retention", "Effective K çarpanı"]]
    for key in ("Q1", "Q2", "Q3", "QUALIFYING_PLAYOFF", "MAIN"):
        rows.append(
            [
                key,
                _num(base[key]),
                _num(retention) if key != "MAIN" else "—",
                _num(eff[key]),
            ]
        )
    return [
        h1("3. Power Elo — maçtan maça güncelleme", s),
        formula(
            [
                "E_home = 1 / (1 + 10 ^ ( -(R_home - R_away + H) / S ))",
                f"  S = {core['elo_scale']:.10f}",
                f"  H = {core['home_advantage']:.10f}   (notr sahada 0)",
                "",
                "Delta_base = K * (S_home - E_home)",
                f"  K = {core['k_factor']:.11f}",
                "",
                "R_home' = R_home + Delta",
                "R_away' = R_away - Delta",
            ],
            s,
        ),
        body(
            "Ev sahibine eklenen miktar deplasmandan birebir düşülür. Bunun sebebi "
            "E_home + E_away = 1 özdeşliğidir: iki takımın beklentisi toplamda bire "
            "eşit olduğu için, gerçekleşen sonucun beklentiden sapması iki takım "
            "için eşit büyüklükte ve ters işaretlidir. Toplam rating kütlesi maç "
            "başına korunur.",
            s,
        ),
        h2("3.1 Eleme turu ağırlıkları", s),
        table(rows, [4.6 * cm, 4.0 * cm, 3.4 * cm, 4.45 * cm], s),
        body(
            "Preliminary Round, Q1 olarak ele alınır. Çarpan her qualifier maçına "
            "gömülüdür; sonradan uygulanan bir düzeltme değildir.",
            s,
        ),
        callout(
            "Ana tura geçişte ne olmaz",
            "Reset yok, carry yok, maç dışı rating değişimi yok. Kulübün ratingi "
            "yalnız maç oynadığında değişir. Doğrudan ana tura giren kulüp için de "
            "taşıma uygulanmaz. Kulüp kimliği kupalar arasında süreklidir: UEL'e "
            "düşen bir kulüp ratingini yanında taşır. Eleme turlarında progression "
            "bonusu verilmez.",
            s,
            tone="amber",
        ),
        page_break(),
    ]


def _section_margin_xg(s: dict, c: dict) -> list[object]:
    gm = c["goal_margin"]
    xg = c["xg_performance"]
    return [
        h1("4. Gol farkı ve xG", s),
        formula(["Delta = Delta_base * GD_multiplier + Delta_xG"], s),
        body(
            "Sıra bağlayıcıdır: önce temel Elo deltası, sonra gol farkı çarpanı, en "
            "son xG düzeltmesi.",
            s,
        ),
        h2("4.1 Gol farkı çarpanı", s),
        formula(
            [
                "GD_multiplier = 1 + alpha * ln(min(GD, cap)) * exp(-|D| / tau)",
                f"  alpha = {_num(gm['alpha'])}",
                f"  tau   = {_num(gm['tau'])}",
                f"  cap   = {_num(gm['goal_difference_cap'])}",
            ],
            s,
        ),
        body(
            "İki kontrol vardır. ln ve cap büyük farkların etkisini sınırlar — 6-0 "
            "ile 4-0 arasında fark yoktur. exp(-|D|/tau) ise favori sönümlemesidir: "
            "güçlü takımın zayıf takımı farklı yenmesi az bilgi taşır, o yüzden az "
            "ödüllendirilir. Beraberlikte ve penaltılarla belirlenen maçta çarpan "
            "1,0'dir, yani etkisizdir.",
            s,
        ),
        h2("4.2 xG performans düzeltmesi", s),
        formula(
            [
                f"Q_xG     = tanh((xG_home - xG_away) / {_num(xg['xg_scale'])})",
                f"Delta_xG = {_num(xg['max_xg_ratio'])} * |Delta_base| * Q_xG",
            ],
            s,
        ),
        table(
            [
                ["Kural", "Davranış"],
                ["İki takımın xG'si birlikte olmalı", "Zorunlu"],
                ["Tek taraflı xG", "Reddedilir; düzeltme uygulanmaz"],
                ["xG eksik", "Yalnız gol farkına düşülür; sıfır performans sayılmaz"],
                ["Beraberlik", "Düzeltme uygulanmaz"],
                ["Penaltı atışları", "Düzeltme uygulanmaz; shootout hariç tutulur"],
                [
                    "Zaman kapsamı",
                    "90 dakika, uzatma varsa eşleşen 120 dakika xG; uymuyorsa "
                    "uygulanmaz",
                ],
                [
                    "Winner-gain korkuluğu",
                    f"minimum oran {_num(xg['minimum_winner_gain_ratio'])}; analitik "
                    "sınır, runtime'da bağlaması beklenmez",
                ],
            ],
            [6.2 * cm, 10.25 * cm],
            s,
        ),
    ]


def _section_draw(s: dict, c: dict) -> list[object]:
    o = c["one_x_two_probability"]
    return [
        h1("5. Beraberlik ve penaltı semantiği", s),
        formula(
            [
                "raw_draw = draw_at_even * (4 * E * (1 - E)) ^ draw_shape",
                "limit    = 2 * min(E, 1 - E)",
                "P_draw   = min(raw_draw, limit)",
                "",
                "P_home = E - 0.5 * P_draw",
                "P_away = 1 - E - 0.5 * P_draw",
                "",
                f"draw_at_even            = {_num(o['draw_at_even'])}",
                f"tek maclik tie          = {_num(o['single_match_draw_at_even'])}",
                f"draw_shape              = {_num(o['draw_shape'])}",
            ],
            s,
        ),
        body("İki özdeşlik her zaman korunur:", s),
        formula(
            [
                "P_home + P_draw + P_away = 1",
                "P_home + 0.5 * P_draw    = E_home",
            ],
            s,
        ),
        callout(
            "İkinci özdeşlik neden kritik",
            "Beraberlik olasılığını değiştirmek Elo beklenen puanını değiştirmez, "
            "yalnız aynı beklentiyi H/D/A arasında farklı dağıtır. Bu yüzden tek maç "
            "draw'ını 0,12'ye indirmek rating state'ini değiştirmez. limit terimi ise "
            "dengesiz maçlarda ham eğrinin negatif olasılık üretmesini engeller.",
            s,
            tone="green",
        ),
        h2("5.1 Penaltı atışları", s),
        body(
            "Penaltı atışlarındaki goller field score'a eklenmez. Bir tie "
            "penaltılarla belirlendiyse, 90 veya 120 dakikadaki skor neyse odur.",
            s,
        ),
        body(
            "Bunun tersi de geçerlidir: shootout, eşit olmayan bir skoru beraberliğe "
            "çevirmez. İlk maçı 2-1 kazanıp turu penaltılarla kaybeden takım için o "
            "maç hâlâ 2-1 galibiyettir.",
            s,
        ),
        page_break(),
    ]


def _section_live(s: dict, c: dict) -> list[object]:
    p = c["progression_bonus"]
    inc = p["increments"]
    caps = p["season_caps"]
    rows = [["Kupa", "Aşama başına", "Sezon tavanı"]]
    for key in ("UCL", "UEL", "UECL"):
        rows.append([key, _num(inc[key]), _num(caps[key])])
    return [
        h1("6. AO Live Elo", s),
        formula(
            ["AO Live Elo = Power Elo + Achievement Reserve + Progression Bonus"],
            s,
        ),
        body(
            "Achievement Reserve aktif değildir; katkısı sabit sıfırdır. Formülde "
            "durur çünkü kapatılabilir bir katmandır, ama bugün hiçbir kulübün "
            "ratingini etkilemez.",
            s,
        ),
        h2("6.1 Progression Bonus", s),
        table(rows, [5.4 * cm, 5.5 * cm, 5.55 * cm], s),
        body(
            "Uygun aşamalar: " + ", ".join(p["eligible_stages"]) + ".",
            s,
        ),
        *bullets(
            [
                "Yalnız son 16 ve sonrası; eleme ve lig aşaması bonus üretmez.",
                "Winner-only: yalnız turu geçen alır.",
                "Tie başına bir kez; iki maçlık tie'da iki kez verilmez.",
                "Kaybedenden düşülmez.",
                "Sezon sonunda sıfırlanır.",
                "Sıfır toplamlı değildir — yukarıdaki iki maddenin doğrudan sonucu.",
            ],
            s,
        ),
        callout(
            "Sıfır toplam bozulmuyor mu",
            "Hayır. Bonus sisteme yeni rating kütlesi ekler, ama Power Elo'nun "
            "dışında, onun üstünde tutulur. Power Elo'nun maç başına sıfır "
            "toplamlılığı bozulmaz.",
            s,
            tone="blue",
        ),
        page_break(),
    ]


def _section_prediction(
    s: dict, c: dict, pred, eff_ao: float, eff_ml: float, eff_poisson: float
) -> list[object]:
    layer = c["prediction_layer"]
    top = layer["top_level_blend"]
    mlc = layer["current_ml_component"]
    poc = layer["ao_domestic_poisson_component"]
    transfer = poc["transfer_config"]
    return [
        h1("7. Servis edilen 1X2 olasılıkları", s),
        body(
            "Olasılık tarafında üç kaynak vardır. Current AO 1X2 ratingi doğrudan "
            "H/D/A'ya çevirir. Structural Logistic, kickoff'tan önce bilinen "
            "özelliklerle eğitilmiş bir lojistik regresyondur. Domestic Poisson, "
            "kulüplerin kendi liglerindeki hücum/savunma parametrelerinden iki gol "
            "beklentisi üretip skor matrisinden H/D/A çıkarır.",
            s,
        ),
        h2("7.1 İç blend'ler", s),
        formula(
            [
                f"Current ML = AO ^ {_num(mlc['ao_weight'])} "
                f"* StructuralLogistic ^ {_num(mlc['ml_weight'])}",
                "",
                f"AO Domestic Poisson = AO ^ {_num(poc['ao_weight'])} "
                f"* Poisson ^ {_num(poc['poisson_weight'])}",
                "",
                f"Served 1X2 = CurrentML ^ {_num(top['current_ml_weight'])} "
                f"* AODomesticPoisson ^ {_num(top['ao_domestic_poisson_weight'])}",
            ],
            s,
        ),
        body(
            f"Birleştirme {top['space']} uzayında yapılır ve sonunda olasılıklar "
            "toplamı bir olacak şekilde normalize edilir. Log uzayında birleştirmenin "
            "anlamı geometrik ortalamadır: bir bileşen bir sonuca çok düşük olasılık "
            "verirse, o sonuç aritmetik ortalamaya göre daha güçlü bastırılır.",
            s,
        ),
        h2("7.2 Efektif katkılar", s),
        table(
            [
                ["Bileşen", "Efektif ağırlık"],
                ["Current AO", f"{eff_ao:.2f}"],
                ["Structural ML", f"{eff_ml:.2f}"],
                ["Domestic Poisson", f"{eff_poisson:.2f}"],
                ["Toplam", f"{eff_ao + eff_ml + eff_poisson:.2f}"],
            ],
            [8.2 * cm, 8.25 * cm],
            s,
        ),
        h2("7.3 Domestic Poisson transfer katsayıları", s),
        formula(
            [
                f"mu                  = {transfer['mu']}",
                f"elo_slope           = {transfer['elo_slope']}",
                f"attack_coefficient  = {transfer['attack_coefficient']}",
                f"defence_coefficient = {transfer['defence_coefficient']}",
                f"rho                 = {_num(transfer['rho'])}",
            ],
            s,
        ),
        body(
            "rho, Dixon-Coles düşük skor düzeltmesidir. Production'da sıfırdır, yani "
            "düzeltme kapalıdır ve skor matrisi bağımsız Poisson'dur. Rho "
            "kalibrasyonu araştırma tarafında mevcuttur ama production'a "
            "taşınmamıştır.",
            s,
        ),
        h2("7.4 Kapsam ve fallback", s),
        table(
            [
                ["Durum", "Davranış"],
                ["Kulüplerin ikisinde de domestic profil var", "BOTH"],
                [
                    "Yalnız birinde var",
                    "ONE — mevcut tarafın profili kullanılır; eksik taraf nötrdür",
                ],
                ["Hiçbirinde yok", "NONE — Poisson bileşeni AO tabanına düşer"],
                [
                    "ML veya Poisson hiç çalışmazsa",
                    f"final = {layer['fallback']}; prediction_status ve "
                    "fallback_reason loglanır",
                ],
                ["Rating state", "Değişmez; prediction rating'e geri beslenmez"],
            ],
            [6.2 * cm, 10.25 * cm],
            s,
        ),
        body(
            "Fallback satırında ML ve Poisson kolonları NaN yazılır, sıfır değil — "
            "\"model çalıştı ve sıfır dedi\" ile \"model çalışmadı\" birbirinden "
            "ayrılır.",
            s,
        ),
        page_break(),
    ]


def _section_monitoring(s: dict) -> list[object]:
    return [
        h1("8. Eksik veri ve izleme", s),
        table(
            [
                ["Sayaç", "Ne ölçer"],
                ["fallback_rate", "FALLBACK_CURRENT_AO durumundaki satırların oranı"],
                [
                    "rows_with_imputed_model_input",
                    "ML çalıştı ama en az bir girdisi impute edildi",
                ],
            ],
            [6.2 * cm, 10.25 * cm],
            s,
        ),
        callout(
            "İki sayaç birbirinin yerine geçmez",
            "İmpute edilmiş girdiyle çalışan bir satır ACTIVE_ENSEMBLE döner. Yani "
            "fallback_rate = 0 olması \"bütün tahminler tam veriyle üretildi\" demek "
            "değildir. Yalnız fallback_rate izlemek, veri kalitesindeki bozulmayı "
            "görünmez kılar.",
            s,
            tone="red",
        ),
        h2("8.1 İzlenmesi gereken sayaçlar", s),
        table(
            [
                ["Sayaç", "Neden"],
                ["fallback_rate", "Bileşen arızasını gösterir"],
                [
                    "rows_with_imputed_model_input",
                    "Veri kalitesi bozulmasını gösterir; fallback'e girmez",
                ],
                [
                    "domestic_poisson_coverage dağılımı",
                    "NONE/ONE oranındaki artış kimlik veya kapsam sorunudur",
                ],
                [
                    "Servis edilen ile Current AO farkı",
                    "Ensemble'ın katkısının izlenmesi",
                ],
                ["Artifact hash uyuşmazlığı", "Zincir bozulması; strict modda durur"],
            ],
            [6.2 * cm, 10.25 * cm],
            s,
        ),
    ]


def _section_layers(
    s: dict,
    c: dict,
    cfg: AOEuropeanEloConfig,
    effective_weights: tuple[float, float, float],
) -> list[object]:
    _, effective_ml, effective_poisson = effective_weights
    ml_label = f"{effective_ml:.2f}".replace(".", ",")
    poisson_label = f"{effective_poisson:.2f}".replace(".", ",")
    rows = [["Katman", "Durum", "Rating'e etkisi", "Prediction'a etkisi"]]
    rows += [
        [
            "Domestic Surprise",
            "ACTIVE",
            f"AO First'ü ±{_num(cfg.domestic_surprise_max_abs_adjustment)} içinde kaydırır",
            "Dolaylı",
        ],
        ["Gol farkı", "ACTIVE", "Power Elo deltasını çarpar", "Dolaylı"],
        ["xG", "ACTIVE", "Deltaya toplanır", "Dolaylı"],
        ["Progression bonus", "ACTIVE", "AO Live'a eklenir", "Dolaylı"],
        ["Kupa katkısı", "ACTIVE", "AO First achievement", "Dolaylı"],
        ["Katılım normalizasyonu", "ACTIVE", "AO First European prior girdisi", "Dolaylı"],
        ["Achievement Reserve", "INACTIVE", "Yok (sabit 0)", "Yok"],
        ["Competition K", "INACTIVE", "Yok; çarpan 1,0", "Yok"],
        ["Dynamic K", "REJECTED", "Yok", "Yok"],
        ["Season power carry", "INACTIVE", "Yok", "Yok"],
        ["Team-specific home context", "SHADOW", "Yok", "Yok"],
        ["Structural ML", "ACTIVE", "Yok", f"{ml_label} efektif ağırlık"],
        ["Domestic Poisson", "ACTIVE", "Yok", f"{poisson_label} efektif ağırlık"],
        [
            "Final ensemble",
            "ACTIVE + izleme",
            "Yok",
            "Servis edilen 1X2'nin kendisi",
        ],
    ]
    return [
        h1("9. Aktif ve aktif olmayan katmanlar", s),
        body(
            "Bir modülün kodda bulunması production'da aktif olduğu anlamına gelmez. "
            "Aşağıdaki tablo contract'ın aktif ilan ettiği katmanları ayırır.",
            s,
        ),
        table(rows, [4.6 * cm, 2.9 * cm, 4.85 * cm, 4.1 * cm], s),
        callout(
            "Son üç satır",
            "Tahmin katmanlarının rating'e etkisinin \"Yok\" olması tesadüf değildir; "
            "sözleşmenin zorunlu tuttuğu bir invariant'tır.",
            s,
            tone="blue",
        ),
        page_break(),
    ]


def _section_evaluation(s: dict) -> list[object]:
    comparison = pd.read_csv(
        ROOT / "reports" / "production_prediction" / "model_comparison.csv"
    ).set_index("model")
    core = pd.read_csv(
        ROOT / "reports" / "current_model" / "model_summary.csv"
    ).set_index("model")
    uncertainty = pd.read_csv(
        ROOT / "reports" / "production_prediction" / "dependency_uncertainty.csv"
    )
    envelope = uncertainty[
        uncertainty["baseline_model"].eq("CURRENT_AO")
        & uncertainty["candidate_model"].eq("ML_POISSON_ENSEMBLE")
        & uncertainty["scope"].eq("ALL")
        & uncertainty["method"].eq("conservative_envelope")
    ]

    def row(label: str, key: str, frame: pd.DataFrame) -> list[str]:
        return [
            label,
            f"{frame.loc[key, 'brier_1x2']:.6f}",
            f"{frame.loc[key, 'log_loss_1x2']:.6f}",
            f"{frame.loc[key, 'accuracy_1x2']:.6f}",
        ]

    ci_rows = [["Metrik", "Fark", "%95 CI alt", "%95 CI üst"]]
    for _, r in envelope.iterrows():
        ci_rows.append(
            [
                str(r["metric"]),
                f"{r['mean_difference']:.6f}",
                f"{r['ci_95_lower']:.6f}",
                f"{r['ci_95_upper']:.6f}",
            ]
        )
    matches = int(comparison.loc["CURRENT_AO", "matches"])
    return [
        h1("10. Güncel değerlendirme sonuçları", s),
        body(
            f"Geliştirme penceresi 2018/19 - 2025/26; altı outer test fold'u "
            f"2020/21 - 2025/26; {matches:,} unseen Avrupa maçı.".replace(",", "."),
            s,
        ),
        h2("10.1 Rating çekirdeği", s),
        table(
            [
                ["Model", "Brier", "Log-loss", "Accuracy"],
                row("Referans çekirdek", "REFERENCE_CORE_NO_ACTIVE_EXTRAS", core),
                row("Current production", "CURRENT_PRODUCTION", core),
            ],
            [6.2 * cm, 3.4 * cm, 3.4 * cm, 3.45 * cm],
            s,
        ),
        h2("10.2 Tahmin katmanı", s),
        table(
            [
                ["Model", "Brier", "Log-loss", "Accuracy"],
                row("Current AO 1X2", "CURRENT_AO", comparison),
                row("Current ML", "CURRENT_ML_BLEND", comparison),
                row("AO Poisson", "AO_POISSON_BLEND", comparison),
                row("Production ensemble", "ML_POISSON_ENSEMBLE", comparison),
            ],
            [6.2 * cm, 3.4 * cm, 3.4 * cm, 3.45 * cm],
            s,
        ),
        h2("10.3 Güven aralıkları", s),
        body(
            "Ensemble ile Current AO arasındaki fark, conservative envelope "
            "yöntemiyle (4.000 bootstrap):",
            s,
        ),
        table(ci_rows, [4.6 * cm, 4.0 * cm, 3.9 * cm, 3.95 * cm], s),
        body(
            "Aynı ensemble'ın kendi ML koluna üstünlüğünde ise aralık sıfırı keser; "
            "yani ML'e karşı kazanç güvenilir değildir.",
            s,
        ),
        callout(
            "Otomatik kapı ile ürün kararı aynı şey değildir",
            "Otomatik gate ensemble'ı KEEP_SHADOW olarak işaretler, çünkü belirsizlik "
            "testini geçemez. Production'da aktif olması manuel bir ürün kararıdır "
            "(PROMOTE_WITH_MONITORING), otomatik bir terfi değildir. İki kaydın "
            "birbirine karıştırılmaması gerekir.",
            s,
            tone="amber",
        ),
        page_break(),
    ]


def _section_limits(s: dict) -> list[object]:
    return [
        h1("11. Kanıt sınırları", s),
        *bullets(
            [
                "Retrospective backtest prospective kanıt değildir; bütün metrikler "
                "geçmiş veriye geriye dönük uygulanmış walk-forward ölçümlerdir.",
                "Değerlendirmedeki ensemble kolu fold bazında seçim yapar; "
                "production sabit %50/%50 kullanır. İkisi aynı sayı değildir.",
                "Güven aralığının sıfırı kesmesi ürün kararını otomatik geçersiz "
                "kılmaz, ama sonucun \"ölçülmüş kazanç\" diye sunulamayacağı "
                "anlamına gelir.",
                "Append-only pre-match ledger olmadan dokunulmamış holdout iddiası "
                "kurulamaz; mevcut prediction çıktısı yeniden üretilebilir bir "
                "dosyadır, tahminin kickoff'tan önce yazıldığını kanıtlamaz.",
                "Provider coverage kapısı eksiksizlik kanıtı değildir; beklenen maç "
                "sayısı moddan veya medyandan çıkarılan bir tahmindir.",
                "Bu repository tek başına canlı bir ingestion servisi değildir.",
            ],
            s,
        ),
        h2("11.1 Hash zinciri neyi kanıtlar, neyi kanıtlamaz", s),
        body(
            "Contract manifest'in hash'ini taşır; manifest her artifact'ın hash'ini "
            "taşır; state dosyaları kendilerini üreten girdi CSV'lerinin hash'lerini "
            "taşır. Zincir artifact değişimini, contract ile manifest'in ayrışmasını "
            "ve state'in farklı bir girdiden üretilmiş olmasını yakalar.",
            s,
        ),
        body(
            "Kanıtlamadığı şey girdinin doğru olduğudur — yalnız ilan edilenle aynı "
            "olduğunu kanıtlar. Sağlayıcı yanlış bir skor döndürdüyse hash zinciri o "
            "yanlış skoru sadakatle mühürler.",
            s,
        ),
    ]


def _section_holdout(
    s: dict, cfg, core: dict, top: dict, mlc: dict, poc: dict
) -> list[object]:
    return [
        h1("12. Holdout protokolü — 8 Eylül 2026 sonrası", s),
        body(
            "Aşağıdaki değerler holdout kilidinden sonra değiştirilemez.",
            s,
        ),
        table(
            [
                ["Parametre", "Donmuş değer"],
                ["max_european_exposure", _num(cfg.max_european_exposure)],
                [
                    "minimum domestic prior ağırlığı",
                    _num(1 - cfg.max_european_exposure),
                ],
                [
                    "katılım normalizasyonu k",
                    _num(cfg.european_participation_shrinkage),
                ],
                ["kupa katkısı w", repr(cfg.cup_contribution_weight)],
                ["bilinmeyen lig sırası", _num(cfg.unknown_league_finish_score)],
                [
                    "Domestic Surprise θ / penalty / cap",
                    f"{_num(cfg.domestic_surprise_coefficient)} / "
                    f"{_num(cfg.domestic_surprise_variance_penalty)} / "
                    f"±{_num(cfg.domestic_surprise_max_abs_adjustment)}",
                ],
                ["elo_scale", f"{core['elo_scale']:.10f}"],
                ["home_advantage", f"{core['home_advantage']:.10f}"],
                ["k_factor", f"{core['k_factor']:.11f}"],
                [
                    "top-level blend",
                    f"{_num(top['current_ml_weight'])} ML / "
                    f"{_num(top['ao_domestic_poisson_weight'])} Poisson",
                ],
                [
                    "ML iç blend",
                    f"{_num(mlc['ao_weight'])} AO / {_num(mlc['ml_weight'])} Structural",
                ],
                [
                    "Poisson iç blend",
                    f"{_num(poc['ao_weight'])} AO / {_num(poc['poisson_weight'])} Poisson",
                ],
            ],
            [7.2 * cm, 9.25 * cm],
            s,
        ),
        body(
            "Ayrıca yasak: kapalı katmanları açmak, feature schema'yı sonuçtan "
            "yararlanacak biçimde genişletmek, eski prediction logunu yeni modelle "
            "yeniden üretip prospective diye sunmak.",
            s,
        ),
        callout(
            "Acil hata düzeltmesi",
            "Bir yazılım hatası güvenlik için düzeltilmek zorundaysa yeni bir "
            "production revision, artifact fingerprint ve etkilenen maç aralığı "
            "açıkça kaydedilir. Eski ve yeni revision sonuçları tek homojen holdout "
            "gibi birleştirilmez.",
            s,
            tone="red",
        ),
        page_break(),
    ]


def _section_operations(s: dict) -> list[object]:
    return [
        h1("13. Canlı işletim ve repo sınırı", s),
        formula(
            [
                "1. fixture ingestion       fikstur listesi ve kickoff'lar",
                "2. identity cozumleme      provider ID -> permanent club_id",
                "3. state checkpoint        domestic Poisson state guncellemesi",
                "4. pre-match feature build kickoff oncesi ozellikler",
                "5. prediction lock         kickoff - 10/15 dk, append-only ledger",
                "6. audit log               status, fallback_reason, fingerprint",
                "7. monitoring              fallback_rate + imputed-input sayaci",
                "8. sonuc replay            mac bitince rating guncellemesi",
            ],
            s,
        ),
        h2("13.1 Repo içinde mevcut", s),
        *bullets(
            [
                "Feature build ve prediction üretimi",
                "Domestic state checkpoint üretimi ve replay",
                "Artifact build ve hash zinciri doğrulaması",
                "Coverage audit ve identity bridge",
                "Prediction log şeması ve audit alanları",
            ],
            s,
        ),
        h2("13.2 Harici servis tarafından sağlanmalı", s),
        *bullets(
            [
                "Fetch worker ve provider rate-limit/retry yönetimi",
                "Kalıcı veritabanı",
                "Identity servisi — birleşme, taşınma ve yeni hukuki kulüp için "
                "manuel karar gerekir",
                "Prediction lock kuyruğu ve append-only ledger",
                "Alarm ve nöbet mekanizması",
                "Read API",
            ],
            s,
        ),
        callout(
            "Belgenin kapsamı",
            "Bu belge modelin ne yaptığını anlatır. Yukarıdaki ikinci liste bu "
            "repository'de mevcut değildir; canlı işletim için ayrıca kurulması "
            "gerekir.",
            s,
            tone="amber",
        ),
        Spacer(1, 0.4 * cm),
        h2("13.3 Kısa sözlük", s),
        table(
            [
                ["Terim", "Tanım"],
                [
                    "AO First Elo",
                    "Sezon başı kulüp gücü; Domestic ve European Prior'ın exposure "
                    "ağırlıklı karışımı",
                ],
                ["Power Elo", "Maçtan maça güncellenen, sıfır toplamlı rating çekirdeği"],
                ["Achievement Reserve", "AO Live formülünde yer alan, bugün kapalı bileşen"],
                [
                    "Progression Bonus",
                    "R16 ve sonrasında turu geçene verilen, sıfır toplamlı olmayan ek",
                ],
                ["AO Live Elo", "Servis edilen kulüp gücü"],
                [
                    "Current AO 1X2",
                    "Ratingten türeyen H/D/A; karşılaştırma tabanı ve fallback",
                ],
                ["Current ML", "AO ile Structural Logistic'in log uzayındaki karışımı"],
                ["AO Domestic Poisson", "AO ile Poisson'un log uzayındaki karışımı"],
                ["Served 1X2", "Nihai servis edilen olasılık"],
            ],
            [4.6 * cm, 11.85 * cm],
            s,
        ),
    ]


if __name__ == "__main__":
    main()
