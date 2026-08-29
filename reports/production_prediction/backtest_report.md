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
| CURRENT_AO | 4884 | 0.566413 | 0.956259 | 0.559173 | 0.000000 | 0.000000 | 0.000000 | 0.002666 | 0.003573 | 0.000614 |
| CURRENT_ML_BLEND | 4884 | 0.563746 | 0.952686 | 0.558559 | -0.002666 | -0.003573 | -0.000614 | 0.000000 | 0.000000 | 0.000000 |
| AO_POISSON_BLEND | 4884 | 0.564488 | 0.953443 | 0.560606 | -0.001925 | -0.002816 | 0.001433 | 0.000741 | 0.000757 | 0.002048 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.564313 | 0.953034 | 0.561016 | -0.002100 | -0.003225 | 0.001843 | 0.000567 | 0.000348 | 0.002457 |
| ML_POISSON_ENSEMBLE | 4884 | 0.563050 | 0.951368 | 0.559582 | -0.003362 | -0.004891 | 0.000410 | -0.000696 | -0.001318 | 0.001024 |

## Fold seçimleri

| fold | test_season | inner_validation_season | poisson_source | ml_weight | poisson_weight |
|---|---|---|---|---|---|
| 1 | 2020/21 | 2019/20 | AO_POISSON_RHO0_CONTROL | 0.400000 | 0.600000 |
| 2 | 2021/22 | 2020/21 | AO_POISSON_BLEND | 0.100000 | 0.900000 |
| 3 | 2022/23 | 2021/22 | AO_POISSON_BLEND | 0.200000 | 0.800000 |
| 4 | 2023/24 | 2022/23 | AO_POISSON_RHO0_CONTROL | 0.700000 | 0.300000 |
| 5 | 2024/25 | 2023/24 | AO_POISSON_RHO0_CONTROL | 0.800000 | 0.200000 |
| 6 | 2025/26 | 2024/25 | AO_POISSON_BLEND | 0.800000 | 0.200000 |

## Unseen fold sonuçları

| fold | test_season | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml | accuracy_1x2 |
|---|---|---|---|---|---|---|---|
| 1 | 2020/21 | 540 | 0.521152 | -0.001717 | 0.889309 | -0.002835 | 0.622222 |
| 2 | 2021/22 | 816 | 0.571021 | -0.001831 | 0.963673 | -0.002204 | 0.549020 |
| 3 | 2022/23 | 804 | 0.575548 | 0.000478 | 0.968796 | -0.000150 | 0.532338 |
| 4 | 2023/24 | 806 | 0.553050 | -0.000576 | 0.936645 | -0.001951 | 0.566998 |
| 5 | 2024/25 | 957 | 0.564938 | -0.001379 | 0.955276 | -0.002052 | 0.557994 |
| 6 | 2025/26 | 961 | 0.575877 | 0.000440 | 0.969667 | 0.000571 | 0.551509 |

## Turnuva ve coverage segmentleri

| segment_type | segment_value | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml |
|---|---|---|---|---|---|---|
| competition | UCL | 1384 | 0.546874 | -0.001558 | 0.927448 | -0.002413 |
| competition | UECL | 2073 | 0.572340 | 0.000465 | 0.964858 | 0.000598 |
| competition | UEL | 1427 | 0.565245 | -0.001545 | 0.954970 | -0.003039 |
| coverage | BOTH | 1443 | 0.581698 | -0.000137 | 0.978174 | -0.000902 |
| coverage | NONE | 1487 | 0.575395 | 0.000155 | 0.968719 | 0.000025 |
| coverage | ONE | 1954 | 0.539885 | -0.001756 | 0.918368 | -0.002648 |

## Kalibrasyon

| model | matches | ece | calibration_slope | calibration_intercept | mean_max_probability |
|---|---|---|---|---|---|
| CURRENT_AO | 4884 | 0.014042 | 1.114606 | 0.037273 | 0.545131 |
| CURRENT_ML_BLEND | 4884 | 0.015624 | 1.035058 | -0.015078 | 0.546522 |
| AO_POISSON_BLEND | 4884 | 0.011704 | 1.044806 | 0.012300 | 0.550562 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.013740 | 1.057397 | 0.031142 | 0.549278 |
| ML_POISSON_ENSEMBLE | 4884 | 0.010635 | 1.051206 | 0.003753 | 0.548948 |

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
