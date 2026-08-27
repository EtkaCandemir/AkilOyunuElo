# Exposure Cap x Katilim Normalizasyonu Etkilesimi

Karar: **`KEEP_CURRENT_CAP`.** Aktif `0.65` korunur. Production contract'a
dokunulmadi.

## Soru

Exposure cap, bazi kuluplerin Avrupa prior'i guvenilmez oldugu icin vardir:
az oynamis bir kulubun bes sezonluk puan toplami az sey ifade eder, cap de
blend'in o sayiya ne kadar yaslanabilecegini sinirlar.

`reports/european_participation/` tam olarak o guvenilmezligin bir kismini
onarir - sezon kaciran kulubun prior'i artik bosluk yuzunden bastirilmaz.
Bu, cap'in ayarini yeniden acar:

- Avrupa sayisi artik daha temizse optimum **yukari** kayabilir
- Ikisi birbirinin yerine geciyorsa **asagi** kayabilir

`0.65`i secen tarama dokunulmamis prior uzerinde kosmustu ve bu soruyu
cevaplayamaz.

## Kurulum

Ayni `14` degerlik cap grid'i iki kez tarandi: bir kez production prior'i
uzerinde, bir kez katilim-normalize prior uzerinde. `k` her cap'te, her
fold'un icinde yeniden secilir - yani iki egri kendi en iyi ayarinda
karsilastirilir, biri digerinden odunc alinmis bir ayarla degil.

## Bulgu 1: iki katman birbirinin yerine gecmiyor

Katilim kazanci cap boyunca neredeyse sabit:

```text
cap    taban Spearman   katilim Spearman   kazanc
0.40      0.456439          0.463699      +0.007260
0.60      0.458608          0.468467      +0.009859
0.65      0.459768          0.469202      +0.009434
0.70      0.460332          0.469187      +0.008855
0.775     0.461323          0.470778      +0.009455
0.85      0.461424          0.470439      +0.009015
```

Aralik `+0.007260` … `+0.009953` (en genis `0.55`te). Katilim duzeltmesi her cap'te ayri bir sey
ekliyor; cap onun yerini tutmuyor.

## Bulgu 2: cap'i yukseltmenin kazanci, katilim duzeltmesi tarafindan yutuluyor

```text
cap 0.65 -> 0.70 ranking kazanci
  taban egrisi        +0.000564
  katilim egrisi      -0.000015
```

Taban egrisinde cap'i yukseltmek ranking kazandiriyordu. Katilim egrisinde
kazandirmiyor. Cap kismen kirli bir Avrupa prior'ini telafi ediyormus; prior
temizlenince telafiye gerek kalmiyor.

## Bulgu 3: hicbir cap `0.65`ten guvenilir sekilde farkli degil

Her cap aktif `0.65`e karsi sezon-blogu bootstrap ile karsilastirildi
(`seed_spearman`, alti fold):

```text
  cap  mean_difference  ci_95_lower  ci_95_upper  reliable
0.400        -0.005501    -0.012233     0.001091     hayir
0.500        -0.002450    -0.006146     0.000949     hayir
0.600        -0.000728    -0.003213     0.001246     hayir
0.675        -0.000064    -0.000354     0.000214     hayir
0.700        -0.000016    -0.000778     0.000680     hayir
0.725        +0.000887    -0.000909     0.003011     hayir
0.750        +0.001091    -0.000673     0.002854     hayir
0.775        +0.001574    -0.000250     0.003646     hayir
0.800        +0.001462    -0.000685     0.003858     hayir
0.825        +0.001385    -0.000979     0.003967     hayir
0.850        +0.001238    -0.001344     0.004139     hayir
```

**Guvenilir sekilde iyi olan cap yok; guvenilir sekilde kotu olan da yok.**
Butun aralik `0.40`-`0.85` boyunca her CI sifiri kesiyor.

Egrinin tepesi `0.775`te (`+0.001574`) fakat CI'si `[-0.000250, +0.003646]`.
Yani egriden tepe okumak bir tercihtir, olcum degildir.

## Karar girdileri

Olcum ayirt edemedigine gore karar baska gerekcelerle verilir, ve ucu de
`0.65`i isaret eder:

**1. Mac loss.** Replay edilen dort cap icinde en iyi Brier `0.65`te:

```text
cap     Brier(katilim)   log-loss    0.65'e gore
0.650      0.568884      0.959528      +0.000000
0.700      0.568976      0.959661      +0.000092
0.775      0.569302      0.960119      +0.000418
0.825      0.569645      0.960626      +0.000761
```

Cap'i yukseltmek ranking'de guvenilir bir sey kazandirmiyor ama loss'tan
gorulebilir miktarda aliyor.

**2. `k` kararliligi.** Foldlar `0.65`te shrinkage uzerinde daha cok anlasir:

```text
cap     k modal pay   k secimleri
0.650      0.67       0.75|0.2|0.2|0.2|0.2|0
0.700      0.33       0.75|0.75|0.2|0.2|0.1|0.1
0.775      0.33       0.75|0.75|0.2|0.2|0.1|0
```

**3. Kanit yoku.** `0.65` aktive edilmis degerdir; degistirmek icin guvenilir
bir fark gerekir ve yoktur.

## Olcek

```text
tum cap boyutunun degeri (0.65 -> 0.775)   +0.0016   guvenilir DEGIL
katilim duzeltmesinin degeri               +0.0094   guvenilir
```

Katilim duzeltmesi, cap'i sonuna kadar optimize etmenin alti katidir ve
istatistiksel olarak ayakta durur. Cap ise bu pencerede ayirt edilemez.

## Sinirlar

- Tek bir alti foldluk pencere. Cap boyutunda `n = 6`; CI'lar bu yuzden genis
  ve "ayirt edilemez" sonucu kismen orneklem buyuklugunun sonucudur.
- Loss ekseni yalniz dort cap icin replay edildi (`0.65`, `0.70`, `0.775`,
  `0.825`); ara degerlerin loss'u bilinmiyor.
- `0.60` katilim kolunda replay edilmedi; taban egrisinde Brier minimumu orada
  oldugu icin katilim kolunda da `0.65`ten iyi olabilir.
- Katilim katmani henuz aktive edilmedi. Aktive edilirse cap sorusu **tekrar**
  sorulmalidir, ve o zaman bu kosu yeniden uretilmelidir.

## Yeniden uretim

```bash
python3 scripts/run_participation_exposure_interaction.py --loss-caps 0.70
```

Yalniz seed ekseni (mac replay'ini atlar):

```bash
python3 scripts/run_participation_exposure_interaction.py --skip-loss-axis
```
