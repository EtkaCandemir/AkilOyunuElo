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
| CURRENT_AO | 4884 | 0.572093 | 0.964371 | 0.550164 | 0.000000 | 0.000000 | 0.000000 | 0.003402 | 0.003913 | -0.001843 |
| CURRENT_ML_BLEND | 4884 | 0.568690 | 0.960458 | 0.552007 | -0.003402 | -0.003913 | 0.001843 | 0.000000 | 0.000000 | 0.000000 |
| AO_POISSON_BLEND | 4884 | 0.569044 | 0.960298 | 0.553235 | -0.003049 | -0.004073 | 0.003071 | 0.000353 | -0.000160 | 0.001229 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.568923 | 0.960066 | 0.552826 | -0.003170 | -0.004304 | 0.002662 | 0.000233 | -0.000391 | 0.000819 |
| ML_POISSON_ENSEMBLE | 4884 | 0.568093 | 0.959242 | 0.553849 | -0.003999 | -0.005129 | 0.003686 | -0.000597 | -0.001216 | 0.001843 |

## Fold seçimleri

| fold | test_season | inner_validation_season | poisson_source | ml_weight | poisson_weight |
|---|---|---|---|---|---|
| 1 | 2020/21 | 2019/20 | AO_POISSON_RHO0_CONTROL | 0.900000 | 0.100000 |
| 2 | 2021/22 | 2020/21 | AO_POISSON_BLEND | 0.000000 | 1.000000 |
| 3 | 2022/23 | 2021/22 | AO_POISSON_BLEND | 0.300000 | 0.700000 |
| 4 | 2023/24 | 2022/23 | AO_POISSON_RHO0_CONTROL | 0.700000 | 0.300000 |
| 5 | 2024/25 | 2023/24 | AO_POISSON_BLEND | 0.800000 | 0.200000 |
| 6 | 2025/26 | 2024/25 | AO_POISSON_BLEND | 0.400000 | 0.600000 |

## Unseen fold sonuçları

| fold | test_season | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml | accuracy_1x2 |
|---|---|---|---|---|---|---|---|
| 1 | 2020/21 | 540 | 0.531981 | -0.000726 | 0.903835 | -0.001005 | 0.594444 |
| 2 | 2021/22 | 816 | 0.572171 | -0.004100 | 0.966765 | -0.004344 | 0.551471 |
| 3 | 2022/23 | 804 | 0.581384 | 0.001008 | 0.977980 | 0.000639 | 0.524876 |
| 4 | 2023/24 | 806 | 0.556752 | -0.001063 | 0.943075 | -0.002933 | 0.564516 |
| 5 | 2024/25 | 957 | 0.568649 | -0.000308 | 0.961743 | -0.001643 | 0.559039 |
| 6 | 2025/26 | 961 | 0.582762 | 0.001209 | 0.979379 | 0.001636 | 0.543184 |

## Turnuva ve coverage segmentleri

| segment_type | segment_value | matches | brier_1x2 | delta_brier_vs_current_ml | log_loss_1x2 | delta_log_loss_vs_current_ml |
|---|---|---|---|---|---|---|
| competition | UCL | 1384 | 0.549387 | -0.001940 | 0.931397 | -0.002967 |
| competition | UECL | 2073 | 0.575785 | 0.000406 | 0.970558 | 0.000344 |
| competition | UEL | 1427 | 0.575062 | -0.000752 | 0.969809 | -0.001783 |
| coverage | BOTH | 1317 | 0.591540 | -0.000615 | 0.993017 | -0.002017 |
| coverage | NONE | 1587 | 0.581285 | 0.000301 | 0.977012 | 0.000418 |
| coverage | ONE | 1980 | 0.541924 | -0.001306 | 0.922533 | -0.001992 |

## Kalibrasyon

| model | matches | ece | calibration_slope | calibration_intercept | mean_max_probability |
|---|---|---|---|---|---|
| CURRENT_AO | 4884 | 0.009766 | 0.927855 | -0.062247 | 0.558511 |
| CURRENT_ML_BLEND | 4884 | 0.015590 | 0.966811 | -0.038749 | 0.550279 |
| AO_POISSON_BLEND | 4884 | 0.006680 | 0.951115 | -0.034192 | 0.553952 |
| AO_POISSON_RHO0_CONTROL | 4884 | 0.008590 | 0.962544 | -0.016847 | 0.552552 |
| ML_POISSON_ENSEMBLE | 4884 | 0.013066 | 0.975041 | -0.026975 | 0.551092 |

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
  "log_loss_fold_wins_vs_current_ml": 4,
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
