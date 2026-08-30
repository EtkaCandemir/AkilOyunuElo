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
| CURRENT_AO | 4884 | 0.565303 | 0.954672 | 0.559582 | 0.000000 | 0.000000 | 0.000000 | 0.003675 | 0.005064 | -0.000205 |
| CURRENT_ML_BLEND | 4884 | 0.561628 | 0.949608 | 0.559787 | -0.003675 | -0.005064 | 0.000205 | 0.000000 | 0.000000 | 0.000000 |
| AO_POISSON_BLEND | 4884 | 0.563527 | 0.952001 | 0.561835 | -0.001776 | -0.002671 | 0.002252 | 0.001899 | 0.002394 | 0.002048 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.563368 | 0.951631 | 0.562039 | -0.001935 | -0.003041 | 0.002457 | 0.001740 | 0.002023 | 0.002252 |
| ML_POISSON_ENSEMBLE | 4884 | 0.561282 | 0.948773 | 0.561220 | -0.004022 | -0.005899 | 0.001638 | -0.000347 | -0.000835 | 0.001433 |

## Fold seçimleri

| fold | test_season | inner_validation_season | poisson_source | ml_weight | poisson_weight |
|---|---|---|---|---|---|
| 1 | 2020/21 | 2019/20 | AO_POISSON_RHO0_CONTROL | 0.500000 | 0.500000 |
| 2 | 2021/22 | 2020/21 | AO_POISSON_BLEND | 0.100000 | 0.900000 |
| 3 | 2022/23 | 2021/22 | AO_POISSON_BLEND | 0.200000 | 0.800000 |
| 4 | 2023/24 | 2022/23 | AO_POISSON_RHO0_CONTROL | 0.700000 | 0.300000 |
| 5 | 2024/25 | 2023/24 | AO_POISSON_RHO0_CONTROL | 0.900000 | 0.100000 |
| 6 | 2025/26 | 2024/25 | AO_POISSON_BLEND | 0.700000 | 0.300000 |

## Unseen fold sonuçları

| fold | test_season | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml | accuracy_1x2 |
|---|---|---|---|---|---|---|---|
| 1 | 2020/21 | 540 | 0.520430 | -0.000976 | 0.888182 | -0.001811 | 0.620370 |
| 2 | 2021/22 | 816 | 0.570163 | -0.001999 | 0.962456 | -0.002420 | 0.552696 |
| 3 | 2022/23 | 804 | 0.574370 | 0.000516 | 0.967059 | -0.000122 | 0.536070 |
| 4 | 2023/24 | 806 | 0.551588 | -0.000618 | 0.934592 | -0.002047 | 0.563275 |
| 5 | 2024/25 | 957 | 0.560607 | -0.000132 | 0.948743 | -0.000246 | 0.561129 |
| 6 | 2025/26 | 961 | 0.574546 | 0.000702 | 0.967824 | 0.000892 | 0.554631 |

## Turnuva ve coverage segmentleri

| segment_type | segment_value | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml |
|---|---|---|---|---|---|---|
| competition | UCL | 1384 | 0.543374 | -0.001097 | 0.922503 | -0.001777 |
| competition | UECL | 2073 | 0.570840 | 0.000736 | 0.962349 | 0.001013 |
| competition | UEL | 1427 | 0.564764 | -0.001191 | 0.954528 | -0.002606 |
| coverage | BOTH | 1443 | 0.579961 | -0.000057 | 0.975696 | -0.000757 |
| coverage | NONE | 1487 | 0.574185 | 0.000559 | 0.967104 | 0.000615 |
| coverage | ONE | 1954 | 0.537667 | -0.001250 | 0.914940 | -0.001996 |

## Kalibrasyon

| model | matches | ece | calibration_slope | calibration_intercept | mean_max_probability |
|---|---|---|---|---|---|
| CURRENT_AO | 4884 | 0.013119 | 1.107724 | 0.036183 | 0.547274 |
| CURRENT_ML_BLEND | 4884 | 0.014909 | 1.063209 | 0.010048 | 0.546909 |
| AO_POISSON_BLEND | 4884 | 0.010388 | 1.047286 | 0.015317 | 0.551656 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.011998 | 1.059269 | 0.033951 | 0.550154 |
| ML_POISSON_ENSEMBLE | 4884 | 0.013995 | 1.071795 | 0.022108 | 0.549266 |

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
