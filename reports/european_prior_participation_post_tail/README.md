# Katilim Shrinkage'i: tail sonrasi yeniden olcum

Karar: **`KEEP_RESEARCH`** — `k = 0.20` degismedi, production'a dokunulmadi.

Bu paket bir **negatif sonucun** kanitidir. Sorulan soru sorulmaya degerdi,
cevap "hayir" cikti, ve ayni soru kilit sonrasi yeniden sorulacagi icin cevap
burada duruyor.

## Soru neden yeniden soruldu

`k = 0.20` zaten bir walk-forward calismasindan geliyordu
(`reports/european_participation/`, `PROMOTE_CANDIDATE`, 2026-08-27). Ama o
calisma `european_tail_beta = 0` altinda kosuldu — yani `european_history_norm`
`1.0`'da sert kesiliyordu.

O kesme onemli bir seyi gizliyordu. Normalizasyon
`rate = gecmis * (1 + k) / (pw + k)` az katilan kulubun gecmisini buyutur;
aktif `k = 0.20`'de carpan `4.44x`'e kadar cikar. Kesme varken doygun
bolgedeki her kulup ayni puana cokuyordu, dolayisiyla sisme siralamada
**gorunmuyordu**. Kesme `2026-08-30`'da kaldirildi
(`production_revision: 2026-08-30-european-prior-tail-no-truncation`) ve sisme
artik siralamada dogrudan ifade buluyor:

```text
kulup              pw    ham gecmis    rate   sisme   EurPrior
SC Freiburg      0.66       13.96     19.47   1.40x    2046.7
Olympique Lyon   0.67       14.87     20.51   1.38x    2072.1
Manchester Utd   0.67       13.90     19.18   1.38x    2039.3
Aston Villa      0.80       23.36     28.04   1.20x    2225.7
```

Aston Villa'nin **ham** Avrupa gecmisi Barcelona (26.51), Inter (26.41) ve
Liverpool'un (26.15) altindadir; normalizasyondan sonra ucunun de ustune
cikar. Opta'nin 2026-08-10 guc siralamasiyla en cok ayristigimiz 12 kulubun
besi tam bu banttadir: Lyon (-20 sira), Freiburg (-18), Celje (-18),
Jagiellonia (-17), Pafos (-10).

Maruziyet terimi bu riskin bir kismini zaten savunuyor — Brighton `3.0x`
sisiyor ama maruziyeti `0.20`. Savunma `pw` 0.60-0.85 bandinda kiriliyor:
orada katilim `0.65` maruziyet tavanini doldurmaya yetiyor ama sismeyi
engellemeye yetmiyor.

## Olcum

Tek eksen; diger her sey aktif production'a pinlendi (`exposure_cap 0.65`,
`european_tail_beta 1.0`, `history_benchmark 20`, `domestic_boost_scale 1.0`,
kalite 1-1). 7 aday, 6 outer fold, 4884 mac, 4000 bootstrap.

```text
   k     Brier      dBrier   log-loss   seed Spearman   d Spearman   ikili dogruluk
0.20   0.567396   0.000000   0.957423       0.467269     0.000000        0.661771
0.35   0.567662  +0.000266   0.957800       0.466614    -0.000655        0.661360
0.50   0.567888  +0.000492   0.958119       0.466115    -0.001154        0.661195
0.75   0.568190  +0.000794   0.958546       0.465968    -0.001301        0.661235
1.00   0.568422  +0.001026   0.958872       0.465498    -0.001770        0.661047
1.50   0.568751  +0.001354   0.959336       0.464683    -0.002586        0.660688
2.50   0.569129  +0.001733   0.959869       0.463211    -0.004058        0.660100
```

Dort olcunun **dordu de** k ile birlikte monoton kotulesiyor ve donus noktasi
yok. Alti outer fold'un altisi da `k = 0.20`'yi secti. Grid `2.50`'ye kadar
uzatildi (orada sisme `1.35x`'e iner) — yani sonuc bir kenar etkisi degil.

Bir yan bulgu: ilk calisma `k`'nin fold'lar arasinda kararsiz oldugunu not
etmisti (modal pay `0.667`, secimler `0.75/0.20/0.20/0.20/0.20/0.00`). Kesme
kalktiktan sonra modal pay **`1.000`** — alti fold'un altisi `0.20`. Yani
truncation'i kaldirmak `k`'yi bozmadi, aksine kararli hale getirdi.

## Yorum

Lyon'un "sismis" gorunmesi bir yanilsama degil, ama duzeltilmesi gereken bir
hata da degil. Lyon'un `20.51`'lik orani, ham `14.87`'sinden **daha iyi** mac
sonucu tahmin ediyor. Az sezon katilmis bir kulubun katildigi sezonlardaki
performansi, katilmadigi sezonlari sifir saymaktan daha bilgilendirici.
Gorunumu duzeltmek dogrulugu bozuyor.

Opta ile ayrisma gercek, ama Opta yer gercegi degil. 4884 macin walk-forward
sonucu yer gercegine daha yakin bir olcu ve o olcu mevcut degeri seciyor.

Sistem seviyesinde bu, maruziyet tavani ile katilim shrinkage'inin **birlikte**
calistigi anlamina geliyor: ince kayitli kulup zaten dusuk maruziyet aliyor,
shrinkage'i ayrica sertlestirmek ayni sinyali iki kez cezalandiriyor.

## Bu kosuda duzeltilen iki kusur

- `BASELINE_CONFIG` bayatti: tail `1.0`'a alindiktan sonra hala `tail = 0`
  tasiyordu. Bundan sonraki her delta, artik servis etmedigimiz bir modele
  gore olculecekti. Artik `AOEuropeanEloConfig.active()`'ten turetiliyor, elle
  yazilmiyor.
- `candidate_count` `81` olarak sabit kodlanmisti. Tail calismasinin
  `selected_candidate.json`'i 28 adaylik bir grid icin `81` yaziyor. Alan
  betimleyicidir, karari etkilemez, ve hicbir izlenen dosyaya (contract,
  reports, docs) ulasmamistir — ama kayit yanlisti. Artik `len(configs)`.

Tail calismasinin kaydi **yeniden uretilmedi**: `BASELINE_CONFIG` artik
tail `1.0` oldugu icin o calismayi yeniden kosmak "tail, tail-oncesi
baseline'i iyilestiriyor" kanitini yok ederdi. O paketteki `81`'in yanlis
oldugu burada kayitlidir.

## Uretim

```bash
python3 scripts/run_european_prior_recalibration_backtest.py \
  --grid participation --bootstrap-samples 4000 \
  --output-root output/european_prior_participation_backtest_2018_2026
```
