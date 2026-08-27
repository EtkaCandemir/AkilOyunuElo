# European Prior: Katilim Normalizasyonu

Karar: **`PROMOTE_CANDIDATE`** -> **`ACTIVE` (2026-08-27)**.

Bu paket aktivasyon kanitidir. Katman `k = 0.20` ile production'a alinmistir
(`production_revision: 2026-08-27-participation-cup-and-unknown-position`).
Asagidaki butun sayilar aktivasyon **oncesi** olculmustur; aktivasyon sonrasi
guncel model metrikleri `reports/current_model/` altindadir.

Aktivasyondan once iki ek kapi kosuldu ve ikisi de gecti:

- `reports/participation_exposure_interaction/` - cap `0.65` kaldi.
- `reports/participation_served_ensemble/` - servis edilen katman guvenli.

## Soru

Dis benchmark modelin olculebilir acigini tek bir yere koyar: **sezon basi
seed'i**. Seed'in icinde ayrimin neredeyse tamamini European Prior tasir
(sd `487`, domestic prior `180`).

Ve o prior sabit agirlikli bir toplamdan gelir:

```python
weighted_european_history = Σ agirlik_k × club_points_k     # k: t-4 … t
```

**Girilmeyen sezon `club_points = 0` katkisi yapar** ve `validators.py:337`
bunu zorunlu kilar. Girilip hic puan alinamayan sezon da sifir katki yapar.
Yani prior su ikisini ayirt edemez:

- "oynadim, kotuydum" -> gercek Avrupa zayifligi
- "katilamadim" -> kanit yoklugu, ve ustelik **domestik** bir olgu

Ayni olgu iki kez faturalandirilir: bir kez prior'da (dusuk history), bir kez
exposure'da (dusuk agirlik). Ikincisi tasarim geregidir ve **bu kosu ona hic
dokunmaz**. Birincisi kontaminasyondur - ve yanlis kanaldadir, cunku
"Avrupa'ya katilabildin mi" domestic prior'in zaten sahiplendigi sorudur.

Etkilenen kutle, `1887` kulup-sezon:

```text
tam katilim (pw = 1.0)      644   34.1%   -> tanim geregi degismez
kismi katilim (0 < pw < 1) 1013   53.7%   -> degisir
hic girmemis (pw = 0)       230   12.2%   -> exposure 0, seed degismez
```

## Formul

```text
pw   = weighted_season_exposure                       # static pipeline zaten yayinliyor
rate = weighted_european_history × (1 + k) / (pw + k)  # pw > 0
rate = 0                                               # pw = 0
```

`(1 + k)` payi guvenligin kaynagidir: `pw = 1` iken `rate` yayinlanmis
history'nin **aynisidir**, yani bes sezonluk tam kaniti olan kulup hic hareket
etmez. Duzeltme yalniz gercek katilim boslugu ile orantilidir.

Form olculerek secildi; elenen alternatifler:

| Form | Tam katilimli 644 kulup |
|---|---|
| `hist / pw` | degismez, ama kuyruk kontrolsuz |
| `hist / (pw + k)` | **medyan -60 Elo** - shrinkage paydasi tam katilimi cezalandiriyor |
| `(hist + k·mu) / (pw + k)` | +10 … +24 Elo - herkesi yukari itiyor |
| **`hist × (1+k) / (pw + k)`** | **tam olarak 0** |

## Uc kol

```text
BASELINE                   production takvim normalizasyonu
PARTICIPATION_BLIND_LIFT   KONTROL - ayni kutle, ayni ortalama hareket,
                           yalniz exposure'in fonksiyonu olarak
PARTICIPATION_NORMALIZED   ADAY
```

Kontrol kolu pazarlik konusu degildir. Onsuz bir kazanc, katilim yapisindan mi
yoksa dusuk exposure'lu kulupleri topluca yukari itmekten mi geldigi ayirt
edilemezdi. Kontrolun katsayisi her fold'da, adayin urettigi ortalama mutlak
seed hareketini esleyecek sekilde egitim sezonlarindan cozulur.

## Sonuc

Sezon basi seed'in gerceklesen performansla Spearman'i, alti fold, `1413`
kulup-sezon:

| Kol | Spearman | Pairwise | Tabana fark |
|---|---:|---:|---:|
| `BASELINE` | 0.459768 | 0.659656 | 0 |
| `PARTICIPATION_BLIND_LIFT` | 0.457303 | 0.658590 | **-0.002465** |
| `PARTICIPATION_NORMALIZED` | **0.469202** | **0.663092** | **+0.009434** |

**Kor kontrol zarar veriyor, aday kazaniyor.** Kazanc katilim yapisindan
geliyor, kutleyi itmekten degil.

### Belirsizlik

```text
seed Spearman, tabana karsi     +0.009462   CI [+0.004316, +0.015733]   GUVENILIR
seed pairwise, tabana karsi     +0.003450   CI [+0.001398, +0.005842]   GUVENILIR
seed Spearman, kontrole karsi   +0.011925   CI [+0.006652, +0.017587]   GUVENILIR
seed pairwise, kontrole karsi   +0.004515   CI [+0.002498, +0.006557]   GUVENILIR
```

### Fold bazinda

Aday hem tabani hem kontrolu **6/6 foldda** gecer:

```text
sezon      taban    kontrol      aday   aday-taban   aday-kontrol
2020/21  0.464493  0.460700  0.467564    +0.003071      +0.006865
2021/22  0.466885  0.466432  0.470184    +0.003299      +0.003752
2022/23  0.456158  0.453330  0.466310    +0.010152      +0.012980
2023/24  0.468141  0.468176  0.490178    +0.022038      +0.022003
2024/25  0.484950  0.480346  0.500012    +0.015063      +0.019667
2025/26  0.417957  0.414826  0.421107    +0.003150      +0.006281
```

### Mac loss ekseni

No-harm kontrolu olarak tasindi, ama zarar vermemekle kalmiyor - **iyilestiriyor**:

| Kol | Brier | log-loss |
|---|---:|---:|
| `BASELINE` | 0.570934 | 0.962432 |
| `PARTICIPATION_BLIND_LIFT` | -0.000083 | -0.000108 |
| `PARTICIPATION_NORMALIZED` | **-0.002051** | **-0.002904** |

```text
Brier     -0.002051   CI [-0.002964, -0.001181]   GUVENILIR
log-loss  -0.002904   CI [-0.004186, -0.001704]   GUVENILIR
kor kontrol: hicbir eksende guvenilir degil
```

Olcek icin: `2026-08-21` exposure cap aktivasyonunun kendi kazanci `-0.00225`
Brier idi. Bu katman onun **ustune** `-0.00205` ekliyor.

### Opta ekseni (diagnostik, tek sezon)

Opta yalniz `2025/26` icin snapshot yayinlar; `236` kulup, karar kapisi degil:

```text
BASELINE                   0.417957   Opta'ya fark -0.068661
PARTICIPATION_BLIND_LIFT   0.414826                -0.071792
PARTICIPATION_NORMALIZED   0.421107                -0.065511
```

Taban degeri `reports/external_benchmark/` ile birebir tutar. Aday, aciğin
yaklasik `%4.6`sini kapatir.

## Guvenlik

Dokuz kapinin tamami gecti:

```text
baseline_reproduces_production   0.0            <- 4.5e-13, servis edilen seed'i birebir uretir
full_participation_unchanged     0.0            <- 644 kulup bit bazinda sabit
zero_participation_unchanged     0.0            <- 230 kulup bit bazinda sabit
control_arm_present              3 kol
candidate_beats_control          1.000          <- 6/6 fold
ranking_not_reliably_harmed      +0.00345
selection_stability              0.667          <- modal k=0.20, 4/6 fold
seed_movement_bounded            97.43 Elo      <- sinir 100
contract_sha256_unchanged        974dfd1f...
```

Nested secim: `k` degerleri fold sirasiyla `0.75 / 0.20 / 0.20 / 0.20 / 0.20 / 0.00`.

## Sinirlar

- **Kutle buyuk.** `%53.7` kulup-sezon hareket ediyor; ortalama `26.6` Elo,
  p95 `97.4`. `seed_movement_bounded` kapisi sinira yakin gecti (`97.43` / `100`).
- **`k` tam kararli degil.** Modal pay `0.667`, kapiyi geciyor ama iki fold
  farkli deger secti. `k = 0` secen fold (`2025/26`) kucuk-payda gurultusune
  aciktir; grid `0`'i bilerek icerir ki bu risk olculebilsin.
- **Opta ekseni tek sezon**, `236` kulup, genis CI. Diagnostik olarak okunmali.
- **Servis edilen ML/Poisson katmani yeniden secilmedi.** Kanit satiri AO Core
  `1X2`'dir; terfi oncesi candidate-seed-aware bir served replay gerekir.
- **Karsi argüman test edildi ve elendi**, ama tek bicimde: kor kontrol
  exposure'in dogrusal bir fonksiyonudur. Baska bir kor bicim farkli sonuc
  verebilir.

## Yeniden uretim

```bash
python3 scripts/run_european_participation_backtest.py
```

Yalniz seed ekseni icin (mac replay'i atlar):

```bash
python3 scripts/run_european_participation_backtest.py --skip-loss-axis
```

Onkosul: dondurulmus seed artifact'i guncel contract ile uyumlu olmalidir.
`baseline_reproduces_production` kapisi bunu her kosuda dogrular; bayat bir
artifact uzerinde kosu `162` Elo hatayla durur.
