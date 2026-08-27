# Domestic Surprise Guclendirme

Karar: **`KEEP_CURRENT`.** Production contract'a dokunulmadi, aktivasyon
yapilmadi. Aktif katman `theta 0.40`, variance penalty `0.50`, domestic cap
`±30`, linear exposure, final cap `±75` olarak kalir.

## Soru

Aktif Domestic Surprise, bir kulubun guncel lig derecesini kendi bes sezonluk
normuyla karsilastirip AO First Elo'yu duzeltir. Bu calisma katsayilarini
buyutmenin - `theta`, domestic cap, exposure ailesi - sezon basi seed'ini
iyilestirip iyilestirmedigini test eder. `206` aday, alti fold, nested secim.

## Sonuc

```text
Pooled Brier farki      -0.000011   CI [-0.000357, +0.000408]   guvenilir degil
Pooled log-loss farki   -0.000032   CI [-0.000504, +0.000520]   guvenilir degil
Fold kazanimi           Brier 4/6, log-loss 4/6
```

Kazanc sifirdan ayirt edilemiyor. Ama asil bulgu kazancin buyuklugu degil,
**nereye kiyasla olctugumuz**.

## Kazanc nereye gitti

Bu calisma iki kez kosuldu. Ilk kosu tabani eski contract'tan (exposure cap
`0.85`) aldi; ikincisi guncel contract'tan (`0.65`) yeniden kurdu:

| Taban | Pooled Brier farki |
|---|---:|
| Eski contract (cap `0.85`) | -0.000555 |
| **Guncel contract (cap `0.65`)** | **-0.000011** |

**53 kat kucullme.** Amplification'in gorunen kazancinin neredeyse tamami,
exposure cap aktivasyonunun zaten yakaladigi seymis. Iki degisiklik ayni yone
calisiyor - domestic kanitin agirligini artirmak - ve cap onu once yapti.

Olcek icin, cap degisikliginin kendi kazanci unseen pencerede `-0.00225` Brier
ve `6/6` fold. Amplification onun ustune `-0.000011` ekliyor: **214 kat fark.**

## Neden `KEEP_CURRENT`

Iki kapi kaliyor.

### 1. Sezon ici ranking guvenilir sekilde bozuluyor

```text
same_season_spearman           -0.002021   CI [-0.003998, -0.000530]   ZARAR
same_season_pairwise_accuracy  -0.001329   CI [-0.002183, -0.000492]   ZARAR
forward_spearman               +0.000293   CI [-0.001123, +0.001623]   -
forward_pairwise_accuracy      -0.000048   CI [-0.000678, +0.000521]   -
```

Iki CI de tamamen sifirin altinda. Katman sifir loss kazancini olculebilir
ranking kaybiyla takas ediyor - ve seed'in var olma sebebi ranking uretmek.

### 2. Foldlar hicbir adayda anlasmiyor

```text
fold 1   theta 0.80   cap 100   vp 0.50   FLOOR_50
fold 2   theta 1.75   cap  30   vp 0.75   POWER_050
fold 3   theta 0.40   cap  60   vp 0.00   LINEAR
fold 4   theta 0.60   cap  75   vp 0.25   LINEAR
fold 5   theta 0.60   cap 150   vp 0.00   LINEAR
fold 6   theta 0.60   cap 150   vp 0.00   LINEAR
```

Alti foldda bes farkli config. `theta` `0.4`-`1.75` (4.4 kat), domestic cap
`30`-`150` (5 kat), variance penalty `0`-`0.75`, uc farkli exposure ailesi.
Modal fold payi `0.333`, kapi `0.50`.

`206` adaylik bir yuzey her zaman fold basina bir kazanan uretir. Kazananlar
birbirini tutmuyorsa pooled loss farki aramanin ozelligidir, kimsenin
gonderebilecegi bir parametre setinin degil.

## Etki nerede yogunlasiyor

```text
exposure bandi   takim-sezon   ort |etki|   p95     maks
0-.20                343         25.22      75.00   75.00
.20-.40              264         24.36      68.42   75.00
.40-.65             1280          8.95      30.13   57.88
```

Kutlenin buyuk etkisi, Avrupa'da en az oynayan kuluplerde: tahmin kalitesine en
az katki yapan ama public ranking'de en gorunur olan yer. Sezon ici Spearman'in
bozulmasinin mekanik aciklamasi budur.

Ayrica `434` takim-sezonu bes sezonluk gecmisi doldurmadigi icin **tam sifir**
alir. Bunlar orantisiz sekilde yeni yukselen ve kucuk lig kulupleri - guncel
formun en bilgilendirici oldugu yer. Katman tam da yardim edecegi yerde susuyor.

## Variance penalty ablation

```text
vp 0.00   en iyi relative loss score  -0.000382
vp 0.25                               -0.000363
vp 0.50   (production)                -0.000329
vp 0.75                               -0.000284
```

Monoton: ceza azaldikca loss iyilesiyor. Yani secim mekanizmasi guvenlik
mekanizmasini sondurmek istiyor. Fakat `variance_penalty`, tek sezonluk bir
flukenin `75` Elo hareket ettirmesini engelleyen tek mekanizmadir; `4884`
maclik loss bir guvenlik parametresi icin dogru amac fonksiyonu degildir.
Grid ayrica dengesiz: `vp=0.5`'te `140` aday, digerlerinde `22`.

## Duzeltilen uc olcum hatasi

Bu paket, calismanin ilk halindeki uc hatanin duzeltilmesinden sonra
uretilmistir. Ucu de ayni yone isaret ediyordu: kanitin desteklemedigi bir
katmani terfi ettirmeye.

**1. Ranking veto'su loss sign convention'ini kullaniyordu.**
`reliable_harm` `lower > 0` ile hesaplaniyordu. Ranking skorlarinda yuksek
iyidir, dolayisiyla zarar negatif farktir ve dogru test `upper < 0`. Yukaridaki
iki guvenilir zarar `False` olarak raporlaniyordu.

**2. Fold secim kararliligi hic olculmuyordu.** Simdi
`fold_selection_modal_share_at_least_half` kapisi var ve her iki terfi
kademesini de veto ediyor.

**3. Kontrol-artifact kontrolu yapisi geregi basarisiz oluyordu.** Dondurulmus
production artifact'i onceki exposure cap ile yazilmisti; cap dusunce cap'e
degen her kulubun domestic agirligi `1-0.85` -> `1-0.65` degisir, dolayisiyla
yeniden kurulan kontrol artifact'e esit **olamaz**. Kontrol ikiye ayrildi:

```text
cap ALTINDA  rebuild artifact'i birebir uretmeli   -> maks fark 5.4e-13   PASS
cap USTUNDE  her sifirdan farkli fark cap'te olmali -> maks fark 6.000000  PASS
```

`6.000000` tam olarak `(0.35 - 0.15) x 30`, yani `±30` domestic cap'indeki bir
kulupte agirlik degisiminin aritmetik sonucu. Bug degil, migration.

Duzeltmeden once karar `AMPLIFY_WITH_MONITORING`'di. Duzeltmeden sonra
`KEEP_CURRENT`. Stored-baseline kolu da `KEEP_CURRENT` veriyordu, ama yanlis
calisan kontrol check'i sayesinde - yani dogru cevaba kazara variyordu.

## Bu kosu neyi kapsamiyor

Servis edilen ML/Poisson katmani yeniden egitilmedi veya yeniden secilmedi.
Kanit satiri AO Core `1X2`'dir. Bazi exposure bantlarinda `25` Elo seed
hareketi yaratan bir degisiklik icin AO Core tek basina yeterli degildir;
terfi oncesi candidate-seed-aware bir served replay gerekir.

## Siradaki adim

Manevra alani amplification'da degil, formulun **girdisinde**:

- Pozisyon percentile yerine lig-sezon ici puan/mac z-score. Veri repoda
  (`data/domestic_league_matches_2013_2026/`, `45.423` mac). Lig buyuklugu
  granularite yanliligini ve "1.-2. farki = 11.-12. farki" sorununu birlikte
  cozer
- Hard cap yerine empirical-Bayes shrinkage (`raw x n/(n+k)`), boylece `434`
  sifir-gecmis takim-sezonu icin ucurum yerine yumusak sinir olur
- Headroom-farkinda surprise: 1. sirada biten kulup iyilesemez, bu yuzden
  `2.` biten sampiyon kalibreli kulup negatif surprise almamali

## Yeniden uretim

```bash
python3 scripts/run_domestic_surprise_amplification_backtest.py \
  --baseline-source rebuilt_current_contract \
  --output-root output/domestic_surprise_amplification_backtest_2018_2026_rebuilt_065
```

Bu paket `rebuilt_current_contract` kolundandir ve otoriter olan odur; stored
kolu yalnizca migration oncesi tabana karsi referans olarak korunur.
