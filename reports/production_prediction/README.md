# Production Prediction Evidence

Bu klasor, aktif `%50 Current ML + %50 AO Domestic Poisson (rho=0)` 1X2
tahmin katmaninin Git'te izlenen kanit paketidir.

## Karar Ayrimi

- `selected_candidate.json`, onceden kayitli otomatik gate'in tarihsel
  `KEEP_SHADOW` kararini degistirilmeden korur.
- `production_activation.json`, kullanicinin daha sonra verdigi operasyonel
  `PROMOTE_WITH_MONITORING` kararini kaydeder.
- Aktivasyon AO Live Elo'yu degistirmez. Tahmin servisinin fallback'i Current
  AO 1X2'dir ve 2026/27 izleme sezonudur.

## Tarihsel Nested Sonuc

Asagidaki `ML_POISSON_ENSEMBLE` kolu, `2018/19-2025/26` gelistirme
penceresindeki `2020/21-2025/26` outer foldlarini degerlendirir. Poisson
kaynagi ve agirligi her foldun onceki sezonlarindan secilir; sabit production
`%50/%50`, `rho=0` politikasinin birebir replay'i degildir. Sayisal kaynaklar
[model_comparison.csv](model_comparison.csv), [fold_results.csv](fold_results.csv)
ve [fold_selections.csv](fold_selections.csv) dosyalaridir.

```text
Unseen mac          4,884
Nested Brier        0.562065
Nested log-loss     0.949965
Nested accuracy     0.561220
Brier farki vs AO  -0.004348
Log farki vs AO    -0.006294
```

Bu paket `2026-08-28` domestic kaynak/fikstur duzeltmesinden sonra yeniden
uretilmistir. AO First ve donmus production parametreleri degismemistir. Servis edilen blend agirligi `--prospective-poisson-weight 0.5`
ile contract degerine pinlenmistir; duzeltilmis veriyle yuzey `0.4` secerdi ve o
tercih `selected_candidate.json` icinde `surface_would_have_selected` olarak
kayitlidir. Blend agirligini yeniden secmek `HOLDOUT_PROTOCOL_2026_27.md` `§5`
tarafindan yasaklanmistir.

28 Agustos runtime bugfix'i sayisal parametreleri degistirmez. Domestic
checkpoint semasi `2.0`, islenmis provider event ID'lerini ve lig cutoff'unu
saklar; eski checkpoint sonuc replay'iyle yenilenir. Duplicate/eski batch
reddedilir ve tahmin gelecekteki lig state'ini kullanamaz. Ayrintili dogrulama:
`reports/production_bugfix_2026_08_28.md`.

Provider sezonu/fikstur onarimi sonrasinda checkpoint 76.327 tekil mac icerir;
61 LIT fiksturunun cift islenmesi kaldirilmistir. 311 eligible kulubun 81'inin
state girdisi degisir; yalniz tekrarlarin etkisi 71 kulupte olculmustur.
Skor kaynaklari ve yeniden uretim kaniti: `reports/domestic_integrity_fix_2026_08_28.md`.

Bu pin yalniz `prospective_2026_27_selection` degerini sabitler. Yukaridaki
nested metrikleri ureten fold agirliklari `0.6/0.9/0.7/0.3/0.1/0.2` olarak
korunur. Current ML'ye karsi fold kazanimi Brier'da `4/6`, log-loss'ta `5/6`dir;
`2024/25` iki metrikte de iyilesmistir. Guven araliklari ve segmentler
[dependency_uncertainty.csv](dependency_uncertainty.csv) ile
[competition_coverage_summary.csv](competition_coverage_summary.csv)
dosyalarindadir. Aktif production karari ve agirliklari degismemistir.

Row-level `unseen_predictions.csv` buyuk ve yeniden uretilebilir oldugu icin
Git'e eklenmez. `scripts/run_final_prediction_ensemble_backtest.py` ile
`output/final_prediction_ensemble_backtest_2018_2026/` altinda uretilir.

Runtime artifact otoritesi:
`artifacts/production_prediction/manifest.json`.
