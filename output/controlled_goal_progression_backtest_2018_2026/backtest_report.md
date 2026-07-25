# Kontrollü Gol Farkı ve Tur Bonusu Backtesti

## Kapsam

- Sezonlar: `2018/19-2025/26`.
- Tarihsel maç: `6340`.
- AO First Elo mimarisi ve sezonluk başlangıç ratingleri değiştirilmedi.
- Scale, H, K, carry=0 ve 1X2 draw mapping mevcut nested fold sözleşmesinden alındı.
- Her dış fold için yeni alpha/tau/base_bonus seçimi yalnızca önceki sezonlarda yapıldı.
- Maçlar kesin UTC sırasıyla işlendi; 2026/27 parametre seçiminde kullanılmadı.
- Penaltıyla biten `155` eşleşmede `S_home=0.5` ve `M_GD=1` uygulandı; bunların `78` tanesinde tek maçın 90/120 skoru eşit değildi ve kullanıcı sözleşmesi bilinçli override edildi.

## Formüller

```text
D = R_home - R_away + H
M_GD = 1 + alpha * ln(min(GD, 4)) * exp(-abs(D) / tau)
Delta = K * (S_home - E_home) * M_GD
```

- Beraberlikte ve penaltı kararında `GD=0`, `M_GD=1`, `S_home=0.5`.
- Progression bonusu eşleşme sonunda bir kez, kazanana `+B` ve elenene `-B` uygulanır.
- UCL/UEL/UECL oranı `1 : 2/3 : 1/3`; qualifying ve lig aşaması bonus dışıdır.

## Out-of-sample Model Karşılaştırması

| Model | Brier | Δ Brier | Log loss | Δ Log | Accuracy | ECE | Rank güvenli |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 0.579490 | +0.000000 | 0.975756 | +0.000000 | 0.5371 | 0.0181 | 5/5 |
| PROGRESSION_ONLY | 0.579477 | -0.000014 | 0.975740 | -0.000016 | 0.5371 | 0.0180 | 3/5 |
| GOAL_DIFFERENCE_ONLY | 0.579369 | -0.000121 | 0.975580 | -0.000176 | 0.5367 | 0.0178 | 3/5 |
| FULL_MODEL | 0.579231 | -0.000260 | 0.975374 | -0.000382 | 0.5371 | 0.0181 | 1/5 |

## İstatistiksel Belirsizlik

- **PROGRESSION_ONLY:** Brier zarfı `[-0.000062, +0.000032]`; log-loss zarfı `[-0.000084, +0.000053]`.
- **GOAL_DIFFERENCE_ONLY:** Brier zarfı `[-0.000296, +0.000022]`; log-loss zarfı `[-0.000438, +0.000032]`.
- **FULL_MODEL:** Brier zarfı `[-0.000455, -0.000084]`; log-loss zarfı `[-0.000671, -0.000121]`.

## Seçilen Geliştirme Adayları

- **PROGRESSION_ONLY:** fixed OOS ranking-first aday `alpha=0, tau=300, base_bonus=12`. Altı nested fold seçimi: `progression_only_a0_t300_b2 | progression_only_a0_t300_b10 | progression_only_a0_t300_b10 | progression_only_a0_t300_b12 | progression_only_a0_t300_b12 | progression_only_a0_t300_b12`.
- **GOAL_DIFFERENCE_ONLY:** fixed OOS ranking-first aday `alpha=0.2, tau=400, base_bonus=0`. Altı nested fold seçimi: `goal_difference_only_a0.2_t500_b0 | goal_difference_only_a0.2_t500_b0 | goal_difference_only_a0_t300_b0 | goal_difference_only_a0_t300_b0 | goal_difference_only_a0_t300_b0 | goal_difference_only_a0_t300_b0`.
- **FULL_MODEL:** fixed OOS ranking-first aday `alpha=0.2, tau=400, base_bonus=4`. Altı nested fold seçimi: `full_model_a0.2_t500_b12 | full_model_a0.2_t500_b6 | full_model_a0.2_t500_b6 | full_model_a0_t300_b12 | full_model_a0_t300_b12 | full_model_a0_t300_b12`.

## Turnuva Segmentleri

| Model | Turnuva | Maç | Δ Brier | Δ Log loss |
| --- | --- | ---: | ---: | ---: |
| FULL_MODEL | UCL | 1384 | -0.000419 | -0.000592 |
| FULL_MODEL | UECL | 2073 | -0.000297 | -0.000459 |
| FULL_MODEL | UEL | 1427 | -0.000051 | -0.000066 |
| GOAL_DIFFERENCE_ONLY | UCL | 1384 | -0.000241 | -0.000346 |
| GOAL_DIFFERENCE_ONLY | UECL | 2073 | -0.000106 | -0.000169 |
| GOAL_DIFFERENCE_ONLY | UEL | 1427 | -0.000025 | -0.000021 |
| PROGRESSION_ONLY | UCL | 1384 | -0.000024 | -0.000019 |
| PROGRESSION_ONLY | UECL | 2073 | -0.000033 | -0.000051 |
| PROGRESSION_ONLY | UEL | 1427 | +0.000025 | +0.000038 |

## Güç Bantları

| Model | Bant | Maç | Δ Brier | Δ Log loss |
| --- | --- | ---: | ---: | ---: |
| FULL_MODEL | BALANCED | 1848 | -0.000085 | -0.000119 |
| FULL_MODEL | HOME_FAVORITE | 2382 | -0.000403 | -0.000596 |
| FULL_MODEL | HOME_UNDERDOG | 654 | -0.000229 | -0.000348 |
| GOAL_DIFFERENCE_ONLY | BALANCED | 1848 | -0.000052 | -0.000072 |
| GOAL_DIFFERENCE_ONLY | HOME_FAVORITE | 2382 | -0.000187 | -0.000276 |
| GOAL_DIFFERENCE_ONLY | HOME_UNDERDOG | 654 | -0.000075 | -0.000108 |
| PROGRESSION_ONLY | BALANCED | 1848 | +0.000012 | +0.000018 |
| PROGRESSION_ONLY | HOME_FAVORITE | 2382 | -0.000019 | -0.000022 |
| PROGRESSION_ONLY | HOME_UNDERDOG | 654 | -0.000064 | -0.000088 |

## Rating Güvenliği

| Model | Min rating | Max rating | Max maç delta | Max sezon hareketi | Elo korunumu |
| --- | ---: | ---: | ---: | ---: | ---: |
| BASE | 623.851 | 2293.307 | 92.636 | 452.375 | 1.164e-10 |
| PROGRESSION_ONLY | 623.851 | 2333.307 | 92.636 | 459.042 | 1.164e-10 |
| GOAL_DIFFERENCE_ONLY | 608.340 | 2293.307 | 92.636 | 472.572 | 1.164e-10 |
| FULL_MODEL | 608.340 | 2351.889 | 92.636 | 512.572 | 1.164e-10 |

## Altı Sorunun Cevabı

1. **Gol farkı Brier ve log-loss'u iyileştiriyor mu?** Nested OOS ΔBrier `-0.000121`, Δlog-loss `-0.000176`; Brier zarfı `[-0.000296, +0.000022]`, log-loss zarfı `[-0.000438, +0.000032]`.
2. **En iyi alpha/tau hangisi?** Geliştirme verisindeki ranking-first sensitivity adayı `alpha=0.2, tau=400`. Nested seçim kararlılığı ve terfi kararı aşağıdaki model kararında dikkate alındı.
3. **Tur bonusu tek başına faydalı mı?** Nested OOS ΔBrier `-0.000014`, Δlog-loss `-0.000016`; Brier zarfı `[-0.000062, +0.000032]`, log-loss zarfı `[-0.000084, +0.000053]`. En iyi sensitivity base_bonus değeri `12`.
4. **Full model anlamlı iyileşiyor mu?** Nested OOS ΔBrier `-0.000260`, Δlog-loss `-0.000382`; Brier zarfı `[-0.000455, -0.000084]`, log-loss zarfı `[-0.000671, -0.000121]`. Sensitivity adayı `alpha=0.2, tau=400, base_bonus=4`.
5. **Fark küçükse sade model mi?** Evet. Promotion için hem dependency zarfının iyileşme yönünde olması hem pratik eşik hem de forward-ranking guardrail'lerinin geçilmesi zorunludur.
6. **Nihai parametre?** Aktif parametreler aşağıdaki veri temelli karar tablosuna göre belirlenmiştir; geçmeyen katmanlar üretim ratingine eklenmez.

## Model Kararı

| Bileşen | Karar |
| --- | --- |
| PROGRESSION_ONLY | **REJECT** |
| GOAL_DIFFERENCE_ONLY | **SHADOW** |
| FULL_MODEL | **SHADOW** |

- Aktif modele alınacak: `yalnızca BASE`.
- Shadow mode: `GOAL_DIFFERENCE_ONLY, FULL_MODEL`.
- Reddedilen: `PROGRESSION_ONLY`.
- Aktif kontrollü katman parametreleri: `alpha=0`, `tau=300` (etkisiz kontrol), `base_bonus=0`.
- AO First Elo ve dondurulmuş klasik canlı çekirdek bu deneyden bağımsızdır.

## Veri ve Metodoloji Sınırları

- Fold'lar bağımsız tekrar değildir; dependency bootstrap sonuçları tie/match, team-season ve calendar-month duyarlılık zarfıdır.
- Fixed-candidate OOS tablosu parametre duyarlılığıdır; dürüst model kararı yalnızca training penceresinde seçilen nested OOS karşılaştırmasına dayanır.
- 2026/27 prospective holdout parametre seçimi için kullanılmayacaktır.
