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
| CURRENT_AO | 4884 | 0.566413 | 0.956259 | 0.559173 | 0.000000 | 0.000000 | 0.000000 | 0.003889 | 0.005356 | -0.001433 |
| CURRENT_ML_BLEND | 4884 | 0.562524 | 0.950903 | 0.560606 | -0.003889 | -0.005356 | 0.001433 | 0.000000 | 0.000000 | 0.000000 |
| AO_POISSON_BLEND | 4884 | 0.564199 | 0.953042 | 0.560606 | -0.002213 | -0.003217 | 0.001433 | 0.001676 | 0.002139 | 0.000000 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.564087 | 0.952790 | 0.561016 | -0.002325 | -0.003469 | 0.001843 | 0.001564 | 0.001887 | 0.000410 |
| ML_POISSON_ENSEMBLE | 4884 | 0.561935 | 0.949792 | 0.561425 | -0.004478 | -0.006468 | 0.002252 | -0.000589 | -0.001112 | 0.000819 |

## Fold seçimleri

| fold | test_season | inner_validation_season | poisson_source | ml_weight | poisson_weight |
|---|---|---|---|---|---|
| 1 | 2020/21 | 2019/20 | AO_POISSON_RHO0_CONTROL | 0.400000 | 0.600000 |
| 2 | 2021/22 | 2020/21 | AO_POISSON_BLEND | 0.100000 | 0.900000 |
| 3 | 2022/23 | 2021/22 | AO_POISSON_BLEND | 0.400000 | 0.600000 |
| 4 | 2023/24 | 2022/23 | AO_POISSON_RHO0_CONTROL | 0.700000 | 0.300000 |
| 5 | 2024/25 | 2023/24 | AO_POISSON_RHO0_CONTROL | 1.000000 | 0.000000 |
| 6 | 2025/26 | 2024/25 | AO_POISSON_BLEND | 0.800000 | 0.200000 |

## Unseen fold sonuçları

| fold | test_season | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml | accuracy_1x2 |
|---|---|---|---|---|---|---|---|
| 1 | 2020/21 | 540 | 0.521320 | -0.001549 | 0.889507 | -0.002637 | 0.620370 |
| 2 | 2021/22 | 816 | 0.570690 | -0.002553 | 0.963739 | -0.002743 | 0.549020 |
| 3 | 2022/23 | 804 | 0.573528 | 0.000053 | 0.965874 | -0.000868 | 0.532338 |
| 4 | 2023/24 | 806 | 0.552386 | -0.000494 | 0.935728 | -0.001949 | 0.565757 |
| 5 | 2024/25 | 957 | 0.562134 | 0.000000 | 0.950922 | 0.000000 | 0.564263 |
| 6 | 2025/26 | 961 | 0.575434 | 0.000416 | 0.969038 | 0.000521 | 0.556712 |

## Turnuva ve coverage segmentleri

| segment_type | segment_value | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml |
|---|---|---|---|---|---|---|
| competition | UCL | 1384 | 0.545742 | -0.001776 | 0.926281 | -0.002712 |
| competition | UECL | 2073 | 0.569959 | 0.000376 | 0.960991 | 0.000375 |
| competition | UEL | 1427 | 0.565983 | -0.000839 | 0.956325 | -0.001719 |
| coverage | BOTH | 1317 | 0.583759 | -0.000926 | 0.982073 | -0.002098 |
| coverage | NONE | 1587 | 0.576784 | 0.000579 | 0.971173 | 0.000610 |
| coverage | ONE | 1980 | 0.535517 | -0.001300 | 0.911182 | -0.001836 |

## Kalibrasyon

| model | matches | ece | calibration_slope | calibration_intercept | mean_max_probability |
|---|---|---|---|---|---|
| CURRENT_AO | 4884 | 0.014042 | 1.114606 | 0.037273 | 0.545131 |
| CURRENT_ML_BLEND | 4884 | 0.015939 | 1.053435 | 0.003446 | 0.546908 |
| AO_POISSON_BLEND | 4884 | 0.010716 | 1.036141 | 0.004431 | 0.551082 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.012233 | 1.048809 | 0.023302 | 0.549916 |
| ML_POISSON_ENSEMBLE | 4884 | 0.012721 | 1.062962 | 0.017308 | 0.548704 |

## 2026/27 için dondurulabilir seçim

- ML ağırlığı: `0.5`
- Poisson ağırlığı: `0.5`
- Poisson kaynağı: `AO_POISSON_RHO0_CONTROL`

Bu tam-geliştirme seçimi 2026/27 verisini görmemiştir; yine de production
aktivasyonu ayrı onay ve prospective doğrulama gerektirir.

## Karar kapıları

```json
{
  "brier_fold_wins_vs_current_ml": 3,
  "log_loss_fold_wins_vs_current_ml": 4,
  "fold_gate": false,
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
