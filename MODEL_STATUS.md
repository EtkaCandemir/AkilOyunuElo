# AO European Elo Model Durumu

Guncelleme tarihi: 2026-07-15

## Kisa karar

AO baslangic Elo modeli ve mac sonrasi dinamik cekirdek, tarihsel arastirma v1
adayi olarak dondurulabilir. Mevcut kanit herhangi bir ana parametreyi degistirmek
icin yeterli ve tutarli degildir.

Bu durum "tum gelecekteki maclarda kanitlanmis production model" anlamina gelmez.
Production kaniti icin model seciminde hic kullanilmamis 2026/27 sezonu gerekir.

## Dondurulan arastirma v1

```text
AO First Elo             = statik v1.1
Dynamic Elo scale        = 225
Home advantage           = 40
Base K                   = 28
Season power carry       = 0.85 (provisional research layer)
UCL forecast scale       = 225
Goal margin multiplier   = inactive
Competition match K      = inactive
Progression bonus        = inactive
Trophy reserve           = inactive
```

`0.85` carry butun Avrupa maclarindaki nested testte guvenilir iyilesme
gosterdigi icin korunur. Harici UCL alt kumesinde `0.50` yon olarak daha iyi
olsa da `0.85` carry'nin zarari guvenilir degildir; bu nedenle daha dar ve
secici bir orneklem ana karari gecersiz kilmaz.

## Kesin tarihli dis benchmark

```text
UEFA exact-date matches          = 6,340 / 6,340
Fresh paired ClubElo matches     = 492
Walk-forward unseen matches      = 363
AO Brier                         = 0.163249
ClubElo Brier                    = 0.158745
AO - ClubElo                     = +0.004504
95% interval                     = [-0.005398, +0.014634]
```

Genel fark kesin degildir. UCL alt kumesinde ClubElo guvenilir bicimde daha
iyidir; UEL yaklasik esittir ve UECL'de AO yon olarak daha iyidir.

## UCL katman teshisi

```text
Static start-only Brier          = 0.175531
Dynamic season-reset Brier       = 0.170063
Current 0.85 carry Brier         = 0.171128
ClubElo Brier                    = 0.156801
AO/ClubElo live rank Spearman    = 0.888094
Current max rating move          = 136.346657
Current minimum rank correlation = 0.907974
```

- Normal dinamik guncelleme baslangic rating'ini iyilestirir.
- `0.85` carry icin UCL'de guvenilir zarar yoktur.
- Tum hareket ve siralama guardrail'leri gecilir.
- UCL olasilik scale adaylari `2/5` fold kazanmistir; iyilesme guvenilir
  olmadigi icin `225` korunur.
- Kalan UCL farki tek bir AO katmanina baglanamamistir. Harici orneklemin daha
  cok guclu ve yerlesik takimlari kapsadigi unutulmamalidir.

## Ne zaman tamamlanmis sayilacak?

### Arastirma v1

Mevcut tarihsel veriyle tamamlanmistir. Parametreler yeni kanit gelene kadar
degistirilmemelidir. Bu surum offline simulasyon, pilot raporlama ve mac sonrasi
puan motorunun kodlanmasi icin kullanilabilir.

### Production dogrulamasi

Asagidaki sozlesme tek seferde, parametre ayarlamadan uygulanmalidir:

1. 2026/27 sezonu bu tarihten sonra untouched holdout olarak kilitlenir.
2. Maclardan once AO tahminleri ve rating'leri degistirilemez loga yazilir.
3. Ilk anlamli ara kontrol lig asamalari bittiginde, 2027 basinda yapilir.
4. Nihai karar 2026/27 Avrupa sezonu tamamlandiginda, 2027 Haziraninda verilir.
5. Brier, log loss, siralama, maksimum rating hareketi ve UCL/UEL/UECL
   segmentleri ayni anda guardrail'leri gecmelidir.

Bu holdout tamamlanmadan model kullanilabilir bir arastirma urunudur; tamamen
kanitlanmis production modeli olarak tanitilmamalidir.

## Ana raporlar

```text
output/external_elo_benchmark_2018_2026/benchmark_report.md
output/ucl_external_diagnostics_2018_2026/diagnostic_report.md
output/ucl_probability_scale_calibration_2018_2026/calibration_report.md
output/achievement_carry_calibration_2018_2026/calibration_report.md
```
