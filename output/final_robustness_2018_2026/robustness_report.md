# AO European Elo Final Robustness

## Karar Ozeti

- Genel durum: `READY_FOR_HOLDOUT_WITH_CAVEATS`
- Nihai comparator: carry=0, exact UTC chronology, nested six-fold core ve training-only 1X2 draw mapping.
- Siralama hedefi, sezon sonu ratingini yalnizca bir sonraki sezonun schedule-adjusted performansiyla karsilastirir.
- Forward siralama, optional katman seciminde Brier ve log-loss'tan once gelir.
- 2026/27 hicbir parametre seciminde kullanilmamistir.

## Optional Katmanlar

### Gol Farki

- Karar: `DISABLE_GOAL_MARGIN`
- Tam veri adayi: `{'family': 'SQRT', 'weight': 1.0, 'cap': 2.0, 'favorite_damping': 0.0}`
- Brier farki: `-0.000403`
- Conservative envelope: `[-0.000984, +0.000176]`
- Brier fold galibiyeti: `3/6`
- Forward siralama gerilemesiz fold: `2/5`
- Iki forward siralama metrigi de iyilesen fold: `0/5`

### Turnuva K

- Karar: `DISABLE_COMPETITION_K`
- Tam veri adayi: `{'profile': 'HIERARCHY', 'ucl_multiplier': 1.5, 'uel_multiplier': 0.75, 'uecl_multiplier': 0.675}`
- Brier farki: `+0.000783`
- Conservative envelope: `[-0.000655, +0.002253]`
- Brier fold galibiyeti: `1/6`
- Forward siralama gerilemesiz fold: `1/5`
- Iki forward siralama metrigi de iyilesen fold: `1/5`

### European Achievement Reserve

- Karar: `DISABLE_ACHIEVEMENT_RESERVE`
- Tam veri adayi: `{'reserve_base': 222.81639928698755, 'uel_multiplier': 0.8, 'uecl_multiplier': 0.6, 'stage_profile': 'LATE_BALANCED', 'reserve_decay': 0.5, 'reserve_cap': 297.0885323826501}`
- Brier farki: `+0.001982`
- Conservative envelope: `[+0.000686, +0.003321]`
- Brier fold galibiyeti: `2/6`
- Forward siralama gerilemesiz fold: `1/5`
- Iki forward siralama metrigi de iyilesen fold: `1/5`

## Final ClubElo Karsilastirmasi

- Durum: `UCL_CLUBELO_POINT_ESTIMATE_BETTER_INCONCLUSIVE`
- External rating bulunan eslesme: `492`
- Walk-forward unseen eslesme: `363`

| Segment | Mac | AO Dynamic | AO Static | ClubElo | AO-ClubElo | Dynamic-Static |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 363 | 0.589487 | 0.606951 | 0.574116 | +0.015371 | -0.017464 |
| UCL | 171 | 0.601615 | 0.606032 | 0.572126 | +0.029489 | -0.004417 |
| UECL | 90 | 0.569702 | 0.621760 | 0.566096 | +0.003606 | -0.052058 |
| UEL | 102 | 0.586612 | 0.595424 | 0.584529 | +0.002083 | -0.008812 |

- UCL 1X2 Brier envelope: `[-0.002932, +0.064568]`
- UCL expected-score MSE envelope: `[-0.000712, +0.030959]`

Bu paired sample ClubElo arsivinin kapsadigi daha yerlesik takimlara agirlik verir; tum on eleme evrenini temsil etmez.
Bu nedenle external sonuc guclu bir diagnostiktir, untouched holdout yerine gecmez.

## Turnuva Residual Diagnostigi

Pozitif residual ev sahibinin model beklentisinden daha fazla puan aldigini, negatif residual daha az aldigini gosterir. Bu tek basina turnuva gucu degildir; yalnizca etiket sonrasinda kalan sistematik kalibrasyon isaretidir.

| Turnuva | Mac | Beklenen | Gerceklesen | Residual | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCL | 1384 | 0.5842 | 0.5770 | -0.0073 | 0.557583 |
| UECL | 2073 | 0.5855 | 0.5907 | +0.0052 | 0.583028 |
| UEL | 1427 | 0.5879 | 0.5837 | -0.0042 | 0.581384 |

## Yorumlama Siniri

Bir katmanin kapali kalmasi fikrin mantiksiz oldugunu degil, bu veri ve onceden tanimli terfi kapilarinda ek tahmin/siralama kaniti uretmedigini ifade eder.
UCL/UEL/UECL K katsayilari turnuva prestiji degil o turnuvadaki maclardan ogrenme hizidir. Prestij gerekirse Power Elo disinda ayri Achievement Index olarak sunulmalidir.
