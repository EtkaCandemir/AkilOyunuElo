# AO ML + Domestic Poisson Final Ensemble Backtesti

## Model kararı

**KEEP_SHADOW**. Bu katman yalnız maç öncesi 1X2 olasılıklarını
birleştirir. AO First Elo, AO Live Elo, Power Delta, gol farkı/xG güncellemesi
ve production contract değiştirilmemiştir.

## Veri ve yöntem

- Geliştirme penceresi: `2018/19-2025/26`, toplam 6.340 Avrupa maçı.
- Unseen değerlendirme: `2020/21-2025/26`, altı fold ve 4.884 maç.
- Her foldun model kaynağı ve ağırlığı yalnız bir önceki inner-validation
  sezonunda seçildi; test sezonu seçime girmedi.
- Karışım log-olasılık uzayında yapıldı. `w=0` mevcut ML, `w=1` Poisson'dur.
- Poisson kaynağında `rho=0` ve inner-selected Dixon-Coles ayrı adaylardır.
- `2026/27` untouched prospective holdout olarak korunmuştur.

## Ana karşılaştırma

| model | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | delta_brier_1x2_vs_ao | delta_log_loss_1x2_vs_ao | delta_accuracy_1x2_vs_ao | delta_brier_1x2_vs_current_ml | delta_log_loss_1x2_vs_current_ml | delta_accuracy_1x2_vs_current_ml |
|---|---|---|---|---|---|---|---|---|---|---|
| CURRENT_AO | 4884 | 0.566413 | 0.956259 | 0.559173 | 0.000000 | 0.000000 | 0.000000 | 0.003663 | 0.005045 | -0.001024 |
| CURRENT_ML_BLEND | 4884 | 0.562750 | 0.951215 | 0.560197 | -0.003663 | -0.005045 | 0.001024 | 0.000000 | 0.000000 | 0.000000 |
| AO_POISSON_BLEND | 4884 | 0.564201 | 0.953045 | 0.560811 | -0.002212 | -0.003214 | 0.001638 | 0.001451 | 0.001831 | 0.000614 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.564089 | 0.952793 | 0.561220 | -0.002323 | -0.003466 | 0.002048 | 0.001339 | 0.001579 | 0.001024 |
| ML_POISSON_ENSEMBLE | 4884 | 0.562065 | 0.949965 | 0.561220 | -0.004348 | -0.006294 | 0.002048 | -0.000685 | -0.001249 | 0.001024 |

## Fold seçimleri

| fold | test_season | inner_validation_season | poisson_source | ml_weight | poisson_weight |
|---|---|---|---|---|---|
| 1 | 2020/21 | 2019/20 | AO_POISSON_RHO0_CONTROL | 0.400000 | 0.600000 |
| 2 | 2021/22 | 2020/21 | AO_POISSON_BLEND | 0.100000 | 0.900000 |
| 3 | 2022/23 | 2021/22 | AO_POISSON_BLEND | 0.300000 | 0.700000 |
| 4 | 2023/24 | 2022/23 | AO_POISSON_RHO0_CONTROL | 0.700000 | 0.300000 |
| 5 | 2024/25 | 2023/24 | AO_POISSON_BLEND | 0.900000 | 0.100000 |
| 6 | 2025/26 | 2024/25 | AO_POISSON_BLEND | 0.800000 | 0.200000 |

## Unseen fold sonuçları

| fold | test_season | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml | accuracy_1x2 |
|---|---|---|---|---|---|---|---|
| 1 | 2020/21 | 540 | 0.521320 | -0.001549 | 0.889508 | -0.002636 | 0.620370 |
| 2 | 2021/22 | 816 | 0.570690 | -0.002554 | 0.963739 | -0.002743 | 0.549020 |
| 3 | 2022/23 | 804 | 0.573972 | 0.000092 | 0.966409 | -0.000652 | 0.532338 |
| 4 | 2023/24 | 806 | 0.552351 | -0.000477 | 0.935675 | -0.001925 | 0.565757 |
| 5 | 2024/25 | 957 | 0.562454 | -0.000537 | 0.951404 | -0.000904 | 0.563218 |
| 6 | 2025/26 | 961 | 0.575434 | 0.000417 | 0.969038 | 0.000521 | 0.556712 |

## Turnuva ve coverage segmentleri

| segment_type | segment_value | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml |
|---|---|---|---|---|---|---|
| competition | UCL | 1384 | 0.546052 | -0.002221 | 0.926748 | -0.003433 |
| competition | UECL | 2073 | 0.570130 | 0.000628 | 0.961331 | 0.000780 |
| competition | UEL | 1427 | 0.565879 | -0.001102 | 0.955971 | -0.002079 |
| coverage | BOTH | 1317 | 0.583951 | -0.000926 | 0.982366 | -0.002126 |
| coverage | NONE | 1587 | 0.576939 | 0.000441 | 0.971422 | 0.000382 |
| coverage | ONE | 1980 | 0.535585 | -0.001427 | 0.911217 | -0.001973 |

## Kalibrasyon

| model | matches | ece | calibration_slope | calibration_intercept | mean_max_probability |
|---|---|---|---|---|---|
| CURRENT_AO | 4884 | 0.014042 | 1.114606 | 0.037273 | 0.545131 |
| CURRENT_ML_BLEND | 4884 | 0.015876 | 1.043793 | -0.000799 | 0.546343 |
| AO_POISSON_BLEND | 4884 | 0.010832 | 1.036090 | 0.004390 | 0.551089 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.012433 | 1.048733 | 0.023230 | 0.549923 |
| ML_POISSON_ENSEMBLE | 4884 | 0.012597 | 1.056235 | 0.013638 | 0.548623 |

## 2026/27 için dondurulabilir seçim

- ML ağırlığı: `0.5`
- Poisson ağırlığı: `0.5`
- Poisson kaynağı: `AO_POISSON_RHO0_CONTROL`

Bu tam-geliştirme seçimi 2026/27 verisini görmemiştir; yine de production
aktivasyonu ayrı onay ve prospective doğrulama gerektirir.

## Karar kapıları

```json
{
  "brier_fold_wins_vs_current_ml": 4,
  "log_loss_fold_wins_vs_current_ml": 5,
  "fold_gate": true,
  "pooled_loss_gate_vs_current_ml": true,
  "uncertainty_gate": false,
  "competition_and_coverage_no_harm_gate": true,
  "calibration_gate": true,
  "probability_gate": true,
  "rating_state_identity_gate": true,
  "beats_ao_brier": true,
  "beats_ao_log_loss": true
}
```
