# AO First Seed Asymmetry Research

Bu paket, ince fakat negatif Avrupa geçmişinin AO First Elo'da yarattığı olası
asimetrinin dar bir boost ile düzeltilip düzeltilemeyeceğini inceler.

## Karar

```text
BOOST_BLIND          KEEP_DIAGNOSTIC_CONTROL
BOOST_DOMESTIC_FORM  KEEP_DIAGNOSTIC
BOOST_LONG_HISTORY   UNAVAILABLE_NOT_RUN
Production change    NONE
```

Blind kol, bağımsız bir futbol sinyali taşımadığı için production adayı
değildir. Domestic-form kolu causal olarak üretildi ancak altı unseen foldun
yalnız birinde ve toplam `1/1413` takım-sezonda rating değiştirdi. Seed Spearman
farkı `+0.000293` olsa da conservative CI `[0.000000, 0.001110]` ile güvenilir
iyileşme göstermedi. 2025/26 Opta farkı da kapanmadı.

## Veri ve yöntem

- Seed hedefi: `1.887` takım-sezon.
- Unseen maç değerlendirmesi: `4.884` UCL/UEL/UECL maçı.
- Outer fold: `2020/21-2025/26`, altı expanding walk-forward sezonu.
- Domestic form: her sezonun ilk UEFA kickoff'undan önce dondurulan causal
  Domestic Poisson attack + defence yüzdeliği.
- Maksimum boost: `150` Elo.
- Sıfır European Exposure: kesinlikle etkilenmez.
- `2026/27`: eğitim, seçim ve test dışında.

Tekrar üretim:

```bash
python3 scripts/run_ao_first_seed_boost_backtest.py --bootstrap-samples 2000
```

Tam ve tekrar üretilebilir tablolar `output/ao_first_seed_boost_backtest_2018_2026/`
altındadır. Bu klasör yalnız review için gerekli küçük kanıt paketidir.

## Dosyalar

- `backtest_report.md`: Türkçe teknik sonuç.
- `model_comparison.csv`: pooled loss ve seed ranking.
- `fold_ranking_results.csv`: unseen fold sonuçları.
- `fold_selections.csv`: yalnız training sezonlarından yapılan seçimler.
- `seed_impact_summary.csv`: müdahale büyüklüğü ve Domestic Surprise etkileşimi.
- `external_opta_comparison.csv`: 2025/26 bağımsız sıralama ekseni.
- `ranking_uncertainty.csv`: cluster ve conservative ranking CI.
- `dependency_uncertainty.csv`: Brier/log-loss no-harm CI.
- `long_history_feasibility.csv`: üçüncü sinyalin neden çalıştırılmadığı.
- `safety_audit.csv`: leakage/invariant/contract kontrolleri.
- `selected_candidate.json`: makine-okunur karar.

