# Forecast-Scale Calibration

Bu çalışma AO Live rating state'ini değiştirmeden, Live Elo farkının H/D/A
olasılıklarına ne kadar keskin çevrileceğini test eder.

- `season_scale_fits.csv`: her sezonun retrospective continuous optimumu.
- `fold_selections.csv`: yalnız geçmiş sezonlardan seçilen forecast scale.
- `fold_results.csv`: altı unseen sezon sonucu.
- `competition_summary.csv`: UCL, UEL ve UECL karşılaştırması.
- `dependency_uncertainty.csv`: bağımlılığa dayanıklı bootstrap aralıkları.
- `selected_candidate.json`: karar ve kapsam.
- `backtest_report.md`: paylaşılabilir ana rapor.

Sonuç `KEEP_SHADOW`dur. Full-history optimumu yaklaşık `906.1` olsa da
training-only seçilen forecast scale pooled 1X2 loss'u iyileştirmemiştir.
Bu nedenle Dynamic Elo `Scale/H/K` production sözleşmesi değiştirilmemiştir.

Yeniden üretim:

```bash
python3 scripts/run_forecast_scale_calibration.py --bootstrap-samples 4000
```
