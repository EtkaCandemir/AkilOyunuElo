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
Served Brier        0.5680932199564515
Served log-loss     0.9592418968185772
Served accuracy     0.5538493038493039
Brier farki vs AO  -0.00399944549914677
Log farki vs AO    -0.00512894042292622
```

Row-level `unseen_predictions.csv` buyuk ve yeniden uretilebilir oldugu icin
Git'e eklenmez. `scripts/run_final_prediction_ensemble_backtest.py` ile
`output/final_prediction_ensemble_backtest_2018_2026/` altinda uretilir.

Runtime artifact otoritesi:
`artifacts/production_prediction/manifest.json`.

