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

## Ana Sonuc

```text
Unseen mac          4,884
Served Brier        0.561935
Served log-loss     0.949792
Served accuracy     0.561425
Brier farki vs AO  -0.004478
Log farki vs AO    -0.006468
```

Bu paket `2026-08-27` revizyonundaki uc statik aktivasyondan (katilim
normalizasyonu, kupa katkisi, bilinmeyen sira tabani) sonra yeniden
uretilmistir. Servis edilen blend agirligi `--prospective-poisson-weight 0.5`
ile contract degerine pinlenmistir; yeni seed altinda yuzey `0.4` secerdi ve o
tercih `selected_candidate.json` icinde `surface_would_have_selected` olarak
kayitlidir. Blend agirligini yeniden secmek `HOLDOUT_PROTOCOL_2026_27.md` `§5`
tarafindan yasaklanmistir.

Row-level `unseen_predictions.csv` buyuk ve yeniden uretilebilir oldugu icin
Git'e eklenmez. `scripts/run_final_prediction_ensemble_backtest.py` ile
`output/final_prediction_ensemble_backtest_2018_2026/` altinda uretilir.

Runtime artifact otoritesi:
`artifacts/production_prediction/manifest.json`.

