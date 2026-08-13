# Degerlendirme ve Kanit Durumu

Bu belge production modelinin son toplu degerlendirmesini ve metodolojik
sinirlarini ozetler. Sayisal otorite:
`reports/current_model/current_model_evaluation_report.md`.

## 1. Evaluation Tasarimi

Gelistirme verisi:

```text
Sezonlar: 2018/19-2025/26
Toplam Avrupa maci: 6,340
Outer test sezonlari: 2020/21-2025/26
Unseen/development evaluation maci: 4,884
Outer fold: 6
```

Expanding walk-forward'da her test sezonu yalniz onceki tamamlanmis sezonlarla
fit edilir. Test sezonu parametre secimine girmez. Bununla birlikte bu pencere
uzun sureli model gelistirmede tekrar kullanildigi icin nihai olarak saf
prospective holdout sayilmaz.

`2026/27` prospective ledger, freeze oncesinde eleme maclari basladigi icin lig
asamasi ve sonrasindan itibaren sonuc gorulmeden once kilitlenecektir.

## 2. Karsilastirma Baseline'i

`REFERENCE_CORE_NO_ACTIVE_EXTRAS`, eski bir production release degildir. Ayni
AO static/dynamic cekirdekte su aktif ekleri kapatan kontrollu ablation'dir:

- Domestic Surprise,
- tek mac beraberlik format duzeltmesi,
- goal margin,
- xG performance,
- progression bonus.

Bu tanim, Scale/H/K veya rating olcegi farkini feature katkisi gibi gostermeyi
engeller.

## 3. Pooled Ana Sonuclar

| Model | Mac | Brier | Log-loss | Accuracy | Spearman | Pairwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AO rating core 1X2 | 4,884 | `0.572093` | `0.964371` | `0.5502` | `0.672160` | `0.753519` |
| Reference core | 4,884 | `0.573699` | `0.967369` | `0.5473` | `0.668059` | `0.752024` |
| Current - reference | | `-0.001606` | `-0.002999` | `+0.0029` | `+0.004101` | `+0.001495` |

Dusuk Brier/log-loss daha iyidir. Current model Brier, log-loss, same-season
Spearman ve pairwise metriklerinde `6/6` fold yon kazanimi uretmistir.

Bu tablo rating motorunun kendi 1X2 ayrisimini degerlendirir. Kullaniciya
sunulan aktif prediction-only ensemble ayri backtestte:

| Model | Mac | Brier | Log-loss | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| AO rating core 1X2 | 4,884 | `0.572093` | `0.964371` | `0.550164` |
| %50 Current ML + %50 AO Domestic Poisson | 4,884 | `0.568093` | `0.959242` | `0.553849` |
| Ensemble - AO | | `-0.003999` | `-0.005129` | `+0.003686` |

Ensemble Current ML'ye karsi Brier ve log-loss'ta `4/6` fold kazanmistir.
Dependency uncertainty otomatik terfi kapisini gecmedigi icin tarihsel karar
`KEEP_SHADOW`dur; operasyonel karar kullanici onayiyla, AO fallback ve 2026/27
izleme kosullari altinda `PROMOTE_WITH_MONITORING` olmustur.

Dependency-robust pooled farklar:

| Metrik | Ortalama fark | %95 CI | Yorum |
| --- | ---: | --- | --- |
| Brier | `-0.001606` | `[-0.002646,-0.000720]` | Guvenilir iyilesme |
| Log-loss | `-0.002999` | `[-0.005207,-0.001124]` | Guvenilir iyilesme |

## 4. Fold Sonuclari

| Test sezonu | Mac | Brier farki | Log-loss farki | Accuracy farki |
| --- | ---: | ---: | ---: | ---: |
| 2020/21 | 540 | `-0.005449` | `-0.012849` | `0.0000` |
| 2021/22 | 816 | `-0.000657` | `-0.001050` | `0.0000` |
| 2022/23 | 804 | `-0.001255` | `-0.001857` | `+0.0050` |
| 2023/24 | 806 | `-0.001041` | `-0.002031` | `0.0000` |
| 2024/25 | 957 | `-0.000766` | `-0.001406` | `+0.0042` |
| 2025/26 | 961 | `-0.001858` | `-0.002471` | `+0.0062` |

## 5. Turnuva Segmentleri

| Turnuva | Mac | Current Brier | Current log-loss | Brier farki | Log-loss farki |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCL | 1,384 | `0.553089` | `0.935921` | `-0.001803` | `-0.003785` |
| UEL | 1,427 | `0.576234` | `0.970406` | `-0.002860` | `-0.005165` |
| UECL | 2,073 | `0.581929` | `0.979211` | `-0.000611` | `-0.000982` |

Uc turnuvada da pooled yon olumludur. UCL mutlak loss olarak en guclu,
UECL en zor segmenttir.

## 6. Feature Ablation

Pozitif `cost_when_disabled`, katman kapatildiginda loss'un arttigini ve aktif
katmanin fayda sagladigini gosterir.

| Kapatilan katman | Brier maliyeti | Log-loss maliyeti |
| --- | ---: | ---: |
| Single-match draw | `+0.000658` | `+0.001617` |
| Domestic Surprise | `+0.000491` | `+0.000722` |
| xG performance | `+0.000242` | `+0.000353` |
| Goal margin | `+0.000194` | `+0.000274` |
| Progression | `-0.0000001` | `-0.0000005` |

Progression prediction loss'a olculebilir katkida bulunmamistir; aktif kalmasi
istatistiksel terfiden cok manuel urun/achievement kararidir.

## 7. AO First Elo Etkisi

Domestic Surprise'in 1,887 takim-sezonundaki etkisi:

| Olcu | Deger |
| --- | ---: |
| Degisen takim-sezon | 1,412 (`%74.8`) |
| Ortalama mutlak AO First farki | `5.68` |
| Medyan mutlak fark | `2.02` |
| P90 | `19.94` |
| P95 | `25.09` |
| Maksimum pozitif | `+30.00` |
| Maksimum negatif | `-28.19` |
| Ortalama mutlak rank degisimi | `2.13` |

Exposure arttikca AO First'e yansiyan Surprise azalir. Ortalama mutlak etki
exposure `0` bandinda `7.90`, `(0.75,1]` bandinda `1.72` Elo'dur.

## 8. Forward Ranking

Bes sezon gecisinde:

```text
Current forward Spearman = 0.468478
Reference               = 0.467063

Current forward pairwise = 0.658162
Reference                = 0.657936
```

Same-season ranking diagnostiktir; forward ranking sezon sonu ratingini bir
sonraki sezon performansiyla iliskilendirir.

## 9. External Initial Elo Validation

2025/26 sezonu baslangic snapshot'i, ilk UEFA macindan onceki `2025-07-03`
Opta club power snapshot'iyla 236/236 takim uzerinde eslestirilmistir:

```text
Spearman          = 0.912757
95% cluster CI    = [0.888453, 0.929452]
Pairwise accuracy = 0.868446
Rank MAE          = 22.322
Decision          = PASS_INITIAL_ELO_EXTERNAL_ALIGNMENT
```

Turnuva segment Spearman:

```text
UCL  0.923814
UEL  0.821074
UECL 0.819373
```

Bu sonuc AO First siralamasinin dis benchmark ile guclu uyumunu gosterir; iki
model ayni hedefi veya veriyi kullanmadigi icin birebir eslesme beklenmez.

Ana dosyalar:

```text
output/initial_elo_external_comparison_2025_26/comparison_summary.csv
output/initial_elo_external_comparison_2025_26/team_comparison.csv
output/initial_elo_external_comparison_2025_26/comparison_report.md
```

## 10. Safety Sonuclari

Son current evaluation'da tum kritik kontroller gecmistir:

- production replay exact match,
- match ve team-season unique keys,
- 1X2 normalization,
- exact 248 single-match tie contract'i,
- match ve sezon Power zero-sum,
- progression cap ve KPO-disabled,
- exposure range,
- insufficient-history zero adjustment,
- Surprise sign/cap,
- chronology ve simultaneous-team collision.

## 11. Production Karari

AO rating engine icin karar: **KEEP**. Kullaniciya sunulan ML + Domestic
Poisson prediction katmani icin karar: **PROMOTE_WITH_MONITORING**.

Bu, modelin calisir, replay edilebilir ve current baseline'dan daha iyi oldugu
anlamina gelir. "Nihai olarak kanitlanmis" anlamina gelmez. En buyuk eksik,
2026/27 prospective locked prediction sonuclaridir. Ensemble rating state'ini
degistirmez; artifact veya feature sorunu Current AO 1X2 fallback'i uretir.

## 12. Raporlama Kurali

Yeni bir test sonunda en az sunlar yazilmalidir:

1. Contract/model fingerprint.
2. Train/test sezonlari ve match count.
3. Baseline tanimi.
4. Pooled Brier/log-loss/accuracy.
5. Fold win ve fold deltalari.
6. UCL/UEL/UECL segmentleri.
7. Cluster CI veya belirsizlik.
8. Rating state identity/conservation.
9. Karar: active, shadow, diagnostic veya reject.
