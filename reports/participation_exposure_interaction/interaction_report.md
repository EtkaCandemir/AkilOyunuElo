# Exposure Cap x Katilim Normalizasyonu Etkilesimi

Production contract degistirilmedi.

## Soru

Cap, bazi kuluplerin Avrupa prior'i guvenilmez oldugu icin vardir: blend'in o
sayiya ne kadar yaslanabilecegini sinirlar. Katilim normalizasyonu tam o
guvenilmezligin bir kismini onarir - sezon kaciran kulubun prior'i artik
bosluk yuzunden bastirilmaz.

Bu, cap'in ayarini yeniden acar. Avrupa sayisi artik daha temizse optimum
yukari kayabilir; yoksa ikisi birbirinin yerine geciyorsa asagi. `0.65`i secen
tarama dokunulmamis prior uzerinde kosmustu ve bu soruyu cevaplayamaz.

## Egri

`k` her cap'te, her fold'un icinde yeniden secilir; iki egri kendi en iyi
ayarinda karsilastirilir.

```text
  cap                      arm  minimum_domestic_weight  teams  seed_spearman  seed_pairwise_accuracy         selected_shrinkages  modal_shrinkage  participation_gain
0.400                 BASELINE                    0.600   1413       0.456439                0.658157                 0|0|0|0|0|0             0.00            0.007260
0.400 PARTICIPATION_NORMALIZED                    0.600   1413       0.463699                0.660899 0.75|0.75|0.35|0.35|0.2|0.2             0.75            0.007260
0.450                 BASELINE                    0.550   1413       0.456486                0.658282                 0|0|0|0|0|0             0.00            0.008764
0.450 PARTICIPATION_NORMALIZED                    0.550   1413       0.465250                0.661524 0.75|0.75|0.35|0.35|0.2|0.2             0.75            0.008764
0.500                 BASELINE                    0.500   1413       0.456864                0.658567                 0|0|0|0|0|0             0.00            0.009883
0.500 PARTICIPATION_NORMALIZED                    0.500   1413       0.466747                0.662334   0.75|0.75|0.2|0.2|0.2|0.2             0.20            0.009883
0.550                 BASELINE                    0.450   1413       0.457619                0.658666                 0|0|0|0|0|0             0.00            0.009953
0.550 PARTICIPATION_NORMALIZED                    0.450   1413       0.467572                0.662613   0.75|0.75|0.1|0.1|0.1|0.1             0.10            0.009953
0.600                 BASELINE                    0.400   1413       0.458608                0.659127                 0|0|0|0|0|0             0.00            0.009859
0.600 PARTICIPATION_NORMALIZED                    0.400   1413       0.468467                0.663051   0.75|0.75|0.1|0.1|0.1|0.1             0.10            0.009859
0.650                 BASELINE                    0.350   1413       0.459768                0.659656                 0|0|0|0|0|0             0.00            0.009434
0.650 PARTICIPATION_NORMALIZED                    0.350   1413       0.469202                0.663092      0.75|0.2|0.2|0.2|0.2|0             0.20            0.009434
0.675                 BASELINE                    0.325   1413       0.460175                0.659806                 0|0|0|0|0|0             0.00            0.008964
0.675 PARTICIPATION_NORMALIZED                    0.325   1413       0.469139                0.663012       0.5|0.5|0.2|0.2|0.1|0             0.50            0.008964
0.700                 BASELINE                    0.300   1413       0.460332                0.659867                 0|0|0|0|0|0             0.00            0.008855
0.700 PARTICIPATION_NORMALIZED                    0.300   1413       0.469187                0.663090   0.75|0.75|0.2|0.2|0.1|0.1             0.75            0.008855
0.725                 BASELINE                    0.275   1413       0.460778                0.660042                 0|0|0|0|0|0             0.00            0.009310
0.725 PARTICIPATION_NORMALIZED                    0.275   1413       0.470088                0.663485   0.75|0.75|0.1|0.1|0.1|0.1             0.10            0.009310
0.750                 BASELINE                    0.250   1413       0.460973                0.660068                 0|0|0|0|0|0             0.00            0.009320
0.750 PARTICIPATION_NORMALIZED                    0.250   1413       0.470293                0.663509   0.75|0.75|0.2|0.2|0.1|0.1             0.75            0.009320
0.775                 BASELINE                    0.225   1413       0.461323                0.660146                 0|0|0|0|0|0             0.00            0.009455
0.775 PARTICIPATION_NORMALIZED                    0.225   1413       0.470778                0.663582     0.75|0.75|0.2|0.2|0.1|0             0.75            0.009455
0.800                 BASELINE                    0.200   1413       0.461289                0.660121                 0|0|0|0|0|0             0.00            0.009378
0.800 PARTICIPATION_NORMALIZED                    0.200   1413       0.470667                0.663575  0.75|0.75|0.75|0.2|0.1|0.1             0.75            0.009378
0.825                 BASELINE                    0.175   1413       0.461510                0.660346                 0|0|0|0|0|0             0.00            0.009079
0.825 PARTICIPATION_NORMALIZED                    0.175   1413       0.470589                0.663624   0.75|0.75|0.1|0.2|0.1|0.1             0.10            0.009079
0.850                 BASELINE                    0.150   1413       0.461424                0.660291                 0|0|0|0|0|0             0.00            0.009015
0.850 PARTICIPATION_NORMALIZED                    0.150   1413       0.470439                0.663504   0.75|0.75|0.2|0.2|0.2|0.1             0.20            0.009015
```

## Tepe noktalari

```text
                     arm   cap  peak_seed_spearman  active_cap_seed_spearman  peak_minus_active  modal_shrinkage
                BASELINE 0.825            0.461510                  0.459768           0.001742             0.00
PARTICIPATION_NORMALIZED 0.775            0.470778                  0.469202           0.001576             0.75
```

Aktif cap `0.65`. Taban egrisinin tepesi
`0.825`, katilim egrisinin tepesi
`0.775` - yani optimum **asagi**
kaydi (`-0.05`).

## Cap boyutunda belirsizlik

Her cap, aktive edilmis `0.65`e karsi sezon-blogu bootstrap
ile karsilastirilir. Egriden tepe okumak bir tercihtir; guvenilir fark ise
olcumdur.

```text
  cap  versus  mean_difference  ci_95_lower  ci_95_upper  reliable_improvement  reliable_harm
0.400    0.65        -0.005501    -0.012233     0.001091                 False          False
0.450    0.65        -0.003950    -0.009216     0.001456                 False          False
0.500    0.65        -0.002450    -0.006146     0.000949                 False          False
0.550    0.65        -0.001624    -0.005147     0.000889                 False          False
0.600    0.65        -0.000728    -0.003213     0.001246                 False          False
0.675    0.65        -0.000064    -0.000354     0.000214                 False          False
0.700    0.65        -0.000016    -0.000778     0.000680                 False          False
0.725    0.65         0.000887    -0.000909     0.003011                 False          False
0.750    0.65         0.001091    -0.000673     0.002854                 False          False
0.775    0.65         0.001574    -0.000250     0.003646                 False          False
0.800    0.65         0.001462    -0.000685     0.003858                 False          False
0.825    0.65         0.001385    -0.000979     0.003967                 False          False
0.850    0.65         0.001238    -0.001344     0.004139                 False          False
```

Aktif cap'ten **guvenilir sekilde iyi** olan cap'ler:
`YOK`.
Guvenilir sekilde kotu olanlar:
`YOK`.

## Mac loss (kisa liste)

```text
  cap                      arm  matches  brier_1x2  log_loss_1x2
0.650                 BASELINE     6340   0.570934      0.962432
0.650 PARTICIPATION_NORMALIZED     6340   0.568884      0.959528
0.700                 BASELINE     6340   0.571176      0.962771
0.700 PARTICIPATION_NORMALIZED     6340   0.568976      0.959661
0.775                 BASELINE     6340   0.571800      0.963678
0.775 PARTICIPATION_NORMALIZED     6340   0.569302      0.960119
0.825                 BASELINE     6340   0.572356      0.964502
0.825 PARTICIPATION_NORMALIZED     6340   0.569645      0.960626
```

## Karar girdisi

- Aktif cap'te katilim kazanci: `+0.009434`
- Kazancin cap boyunca araligi:
  `+0.007260` … `+0.009953`

Karar urun tarafina aittir; bu belge yalniz kanit uretir.
