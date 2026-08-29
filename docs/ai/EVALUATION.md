# Degerlendirme ve Kanit Durumu

Bu belge production modelinin son toplu degerlendirmesini ve metodolojik
sinirlarini ozetler. Sayisal otorite:
`reports/current_model/current_model_evaluation_report.md`.

28 Agustos 2026 revision'u parametre retuning'i degildir. On giris/state hatasi
regresyon senaryolariyla ele alinmistir; bu sentetik kontrollerin gecmesi yeni
bir performans veya prospective holdout kaniti sayilmaz. Domestic checkpoint
`2.0` gecisi sonuc gecmisinden yeniden uretim gerektirir. Dogrulama komutlari
ve yeniden uretilen artifact kanitlari `reports/production_bugfix_2026_08_28.md`
icinde tutulur.

Ayni gunun domestic veri butunlugu duzeltmesi ayrica
`reports/domestic_integrity_fix_2026_08_28.md` icinde izlenir. Onceki checkpoint
61 adet LIT 2026 fiksturunu history ve live tarafinda iki kez isliyordu.
Provider sezonu ve coverage duzeltmesinden sonra veri/state ve etkilenen
tahmin kaniti yeniden uretilir; sadece hash yenilemek yeterli degildir.
Bu duzeltme ertelenen UTC fold-siniri bulgusunu veya prospective ledger eksigini
kapatmaz; tarihsel testlere untouched holdout denemez.

Tarihsel replay metrikleri 18 Agustos 2026 continuous qualifier-retention
aktivasyonundan once uretilmistir. Aktif runtime Q1/Q2/Q3/QPO icin
`0.20/0.275/0.35/0.425`, MAIN icin `1.00` efektif K carpanlarini kullanir ve
MAIN girisinde carry/reset uygulamaz. Bu degisiklik 2026/27 prospective
ledger'da ayri revision olarak izlenecektir.

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
| AO rating core 1X2 | 4,884 | `0.566413` | `0.956259` | `0.5592` | `0.683258` | `0.759421` |
| Reference core | 4,884 | `0.568053` | `0.959174` | `0.5545` | `0.681487` | `0.758195` |
| Current - reference | | `-0.001640` | `-0.002915` | `+0.0047` | `+0.001771` | `+0.001226` |

Dusuk Brier/log-loss daha iyidir. Current model Brier ve log-loss'ta `6/6`,
same-season Spearman ve pairwise metriklerinde `6/6` fold yon kazanimi
uretmistir. Kaynak [fold_summary.csv](../../reports/current_model/fold_summary.csv):
`CURRENT_PRODUCTION`, ayni foldun `REFERENCE_CORE_NO_ACTIVE_EXTRAS` satiriyla
karsilastirilir; iki ranking metriginin de alti farki pozitiftir.

Referans kolu da aktif `0.65` exposure cap'iyle seed'lenir. Onceki yayimlanmis
referans (`0.573699`, fark `-0.004411`) bayat bir static manifest uzerinden
`0.85` cap ile uretiliyordu; bu, `§2`'nin acikca engellemek istedigi seyi
yapiyor ve exposure kararinin kendi kazancini feature katkisi gibi
gosteriyordu. Manifest duzeltildikten sonra `CURRENT_PRODUCTION` degerleri
degismedi; yalniz referans ve no-surprise kollari duzeldi.

Bu tablo rating motorunun kendi 1X2 ayrisimini degerlendirir. Prediction-only
ensemble'in ayri **nested walk-forward** backtesti asagidadir. Her foldun
Poisson kaynagi ve ust blend agirligi yalniz o foldun onceki sezonlarindan
secilir; bu tarihsel kol, production'daki sabit `%50/%50`, `rho=0` politikasinin
birebir replay'i degildir.

| Model | Mac | Brier | Log-loss | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| AO rating core 1X2 | 4,884 | `0.566413` | `0.956259` | `0.559173` |
| Nested ML + Poisson (`ML_POISSON_ENSEMBLE`) | 4,884 | `0.562065` | `0.949965` | `0.561220` |
| Ensemble - AO | | `-0.004348` | `-0.006294` | `+0.002048` |

Iki tablonun AO kolu aynidir (`0.566413`). Guncel sayilar, `2026-08-28` domestic kaynak/fikstur duzeltmesinden sonraki
[model_comparison.csv](../../reports/production_prediction/model_comparison.csv)
dosyasindan gelir. Katilim gate'inin ayri
[aday snapshot'i](../../reports/participation_served_ensemble/model_comparison_candidate.csv)
`0.562152` Brier ve AO'ya `-0.004360` fark tasir; guncel nested kolun degerleri
`0.562065` ve `-0.004348`dir. Iki snapshot birbirinin yerine kullanilmaz.

Nested ensemble Current ML'ye karsi Brier'da `4/6`, log-loss'ta `5/6` fold
kazanmistir; `2024/25` iki metrikte de iyilesmistir. Kaynak
[fold_results.csv](../../reports/production_prediction/fold_results.csv).
Dependency uncertainty otomatik terfi kapisini gecmedigi icin tarihsel karar
`KEEP_SHADOW`dur; operasyonel karar kullanici onayiyla, AO fallback ve 2026/27
izleme kosullari altinda `PROMOTE_WITH_MONITORING` olmustur.

Nested kolda Poisson agirliklari fold sirasiyla `0.6/0.9/0.7/0.3/0.1/0.2`dir;
kaynak da foldlar arasinda degisir
([fold_selections.csv](../../reports/production_prediction/fold_selections.csv)).
`--prospective-poisson-weight 0.5` yalniz ileriye donuk secimi pinler; bu
tarihsel satirlari sabit agirlikla yeniden hesaplamaz. Aktif production
politikasi `%50 Current ML + %50 AO Domestic Poisson (rho=0)` olarak korunur.

Nested ensemble icin dependency-robust pooled farklar
([dependency_uncertainty.csv](../../reports/production_prediction/dependency_uncertainty.csv),
`scope=ALL`, `method=conservative_envelope`):

| Baseline | Brier farki [%95 CI] | Log-loss farki [%95 CI] |
| --- | --- | --- |
| Current AO | `-0.004348 [-0.006442,-0.002186]` | `-0.006294 [-0.009636,-0.002765]` |
| Current ML | `-0.000685 [-0.001662,+0.000218]` | `-0.001249 [-0.002901,+0.000265]` |

AO rating core - reference core icin dependency-robust pooled farklar:

| Metrik | Ortalama fark | %95 CI | Yorum |
| --- | ---: | --- | --- |
| Brier | `-0.001640` | `[-0.002905,-0.000538]` | Guvenilir iyilesme |
| Log-loss | `-0.002915` | `[-0.005270,-0.000946]` | Guvenilir iyilesme |

## 4. Fold Sonuclari

| Test sezonu | Mac | Brier farki | Log-loss farki | Accuracy farki |
| --- | ---: | ---: | ---: | ---: |
| 2020/21 | 540 | `-0.006610` | `-0.013642` | `+0.0037` |
| 2021/22 | 816 | `-0.000565` | `-0.000950` | `-0.0012` |
| 2022/23 | 804 | `-0.003059` | `-0.004303` | `+0.0050` |
| 2023/24 | 806 | `-0.000243` | `-0.000908` | `0.0000` |
| 2024/25 | 957 | `-0.000343` | `-0.000687` | `+0.0021` |
| 2025/26 | 961 | `-0.001036` | `-0.001293` | `+0.0166` |

Brier ve log-loss `6/6` foldda olumlu yondedir. Accuracy `4/6` foldda
olumlu, birinde notr, 2021/22'de hafif negatiftir.

## 5. Turnuva Segmentleri

| Turnuva | Mac | Current Brier | Current log-loss | Brier farki | Log-loss farki |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCL | 1,384 | `0.550058` | `0.932333` | `-0.002542` | `-0.004804` |
| UEL | 1,427 | `0.566628` | `0.956206` | `-0.002455` | `-0.004253` |
| UECL | 2,073 | `0.577183` | `0.972269` | `-0.000477` | `-0.000732` |

Uc turnuvada da pooled yon olumludur. UCL mutlak loss olarak en guclu,
UECL en zor segmenttir.

## 6. Feature Ablation

Pozitif `cost_when_disabled`, katman kapatildiginda loss'un arttigini ve aktif
katmanin fayda sagladigini gosterir.

| Kapatilan katman | Brier maliyeti | Log-loss maliyeti | Brier %95 CI | Guvenilir zarar |
| --- | ---: | ---: | --- | :--: |
| Single-match draw | `+0.000714` | `+0.001602` | `[+0.000003,+0.001688]` | evet |
| xG performance | `+0.000674` | `+0.000971` | `[+0.000036,+0.001369]` | evet |
| Domestic Surprise | `+0.000132` | `+0.000186` | `[-0.000189,+0.000401]` | hayir |
| Goal margin | `+0.000061` | `+0.000079` | `[-0.000126,+0.000234]` | hayir |
| Kupa katkisi | `+0.000022` | `+0.000041` | `[-0.000202,+0.000217]` | hayir |
| Progression | `-0.0000003` | `-0.0000008` | `[-0.000006,+0.000005]` | hayir |

Bu tablo alti sezonluk xG haritasiyla ve her iki kolda ayni `0.65` exposure
cap'iyle uretilmistir. Iki eski kopya sinifi stale kabul edilmelidir: tek
sezonluk xG haritasiyla uretilenlerde xG `+0.000242` gorunur, bayat `0.85`
referans seed'iyle uretilenlerde Domestic Surprise `+0.002726` ile birinci
sirada gorunur. Ikinci sayinin `+0.00225`lik fazlasi exposure cap kararinin
kendi kazancidir, katmanin degeri degildir.

Brier ekseninde kapatilmasi guvenilir zarar veren **iki** katman vardir: tek
mac beraberlik format duzeltmesi ve bounded xG. Ikisinde de log-loss ekseni
ayni sonucu vermez; tek mac katmaninin log-loss CI'si sifiri keser.

**Tek mac beraberlik katmani icin onemli bir kanit siniri vardir ve bu sinir
yukaridaki "evet" isaretinden daha belirleyicidir.** Katmanin ornekleminde
`224` tek mac vardir ve bunlarin `200`'u (`%89`) yalniz `2020/21` COVID
sezonundan gelir; COVID sonrasi segment (`2021/22+`) yalnizca `24` mactir.
Brier CI'sinin alt siniri `+0.000003`, yani sifiri yedinci hanede keser -
mumkun olan en zayif "guvenilir" verdikt. COVID sonrasi segment ayni yonu
gosterir (`-0.016` Brier) ama `24` macla. Bu katmani `2827` uygun maca
dayanan bounded xG ile ayni guven sinifinda sunmak yaniltici olur; format
duzeltmesi yapisal gerekceyle aktiftir, orneklem gucuyle degil. Segment
ayrintisi `reports/single_match_draw_backtest/segment_summary.csv`
icindedir.

Domestic Surprise'in maliyeti pozitif fakat CI'si sifiri kesiyor; katman
`ACTIVE` kalir cunku aktivasyonu zaten manuel urun karariydi, ancak
"olculebilir en degerli katman" iddiasi gecerli degildir. Katilim
normalizasyonu aktive edilince Surprise'in maliyeti `+0.000287` ->
`+0.000132`ye iner: iki katman kismen ayni kutleye - Avrupa kaniti ince
kulupler - dokunur, dolayisiyla katilim duzeltmesi Surprise'in tasidigi isin
bir kismini ustlenir.

Kupa katkisi genellestirildikten sonra o kolun maliyeti **isaret degistirdi**:
duble bonusu doneminde `-0.000080` (kapatmak iyilestiriyordu) iken simdi
`+0.000022` (kapatmak hafifce kotulestiriyor). Fark hala guvenilir degildir,
ama katmanin artik modele karsi degil lehine calistigini gosterir.

Progression prediction loss'a olculebilir katkida bulunmamistir. Ayrintili
tartisma `RESEARCH_STATUS` `§6`'dadir.

## 7. AO First Elo Etkisi

Domestic Surprise'in 1,887 takim-sezonundaki etkisi:

| Olcu | Deger |
| --- | ---: |
| Degisen takim-sezon | 1,412 (`%74.8`) |
| Ortalama mutlak AO First farki | `6.58` |
| Medyan mutlak fark | `3.68` |
| P90 | `19.94` |
| P95 | `25.09` |
| Maksimum pozitif | `+30.00` |
| Maksimum negatif | `-28.19` |
| Ortalama mutlak rank degisimi | `2.29` |

Exposure arttikca AO First'e yansiyan Surprise azalir. Ortalama mutlak etki
exposure `0` bandinda `7.90`, `(0.50,0.75]` bandinda `4.29` Elo'dur. Aktif
cap `0.65` oldugu icin `(0.75,1]` bandi bostur.

## 8. Forward Ranking

Bes sezon gecisinde:

```text
Current forward Spearman = 0.475864
Reference                = 0.472581

Current forward pairwise = 0.661164
Reference                = 0.659794
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

### 9.1 Guncel Model Dis Benchmark

Yukaridaki karsilastirma yalniz *uyumu* olcer ve mac sonuclarini kullanmaz.
Aktif contract'in dis referanslara karsi gercek performansi ayri bir pakette
olculur:

```bash
python3 scripts/run_current_external_benchmark.py
```

Eksen 1, servis edilen 1X2'yi ClubElo'ya karsi puanlar. `363` eslesmis mac,
snapshot yasi `<=31` gun, ClubElo ev sahibi avantaji yalniz onceki sezonlardan
fit edilir ve iki tarafa ayni beraberlik modeli uygulanir:

```text
Climatology (walk-forward)   Brier 0.642983  log-loss 1.062634
ClubElo (yayinlanmis 400)    Brier 0.574983  log-loss 0.966819
AO rating cekirdegi          Brier 0.577244  log-loss 0.972194
AO servis edilen ensemble    Brier 0.579166  log-loss 0.973577
```

ClubElo nokta tahmininde hala ondedir fakat acik katilim normalizasyonu
aktivasyonuyla belirgin sekilde daralmistir: AO cekirdeginin ClubElo'ya
Brier farki `+0.012852` -> `+0.002261`, servis edilen kolunki `+0.010958`
-> `+0.004183` (28 Agustos domestic veri onarimi sonrasi). Dort karsilastirmanin dordunde de conservative zarf sifiri
keser; yani hicbir yonde guvenilir fark yoktur.

Eksen 2, sezon basi ratingleri gerceklesen 2025/26 performansina karsi puanlar.
Hedef, hicbir ratingden etkilenmeyen leave-team-out schedule-adjusted skordur:

```text
rating                            reference_type  Spearman  pairwise
AO First Elo                      MODEL           0.423272  0.647167
AO First Elo, Surprise kapali     MODEL_ABLATION  0.421493  0.646262
Opta pre-season                   EXTERNAL        0.486618  0.671870
UEFA kulup katsayisi (pre-season) OWN_INPUT       0.268334  0.596310
```

Eslesmis Spearman farklari:

```text
AO - Opta                        -0.063319   95% CI [-0.106158, -0.020041]  Opta guvenilir
AO - UEFA katsayisi              +0.153808   95% CI [+0.056536, +0.257488]  AO guvenilir
AO - AO(Surprise kapali)         +0.001779   95% CI [-0.004942, +0.008528]  guvenilir degil
AO(Surprise kapali) - Opta       -0.065098   95% CI [-0.107604, -0.022454]  Opta guvenilir
```

#### Domestic Surprise seed ekseninde

`MODEL_ABLATION` kolu, katmanin gerekcesi olan ekseni olcer: mac tahmini degil,
sezon basi seed kalitesi. Katman `236` takimin `181`ini oynatir, ortalama
mutlak hareket `8.90` Elo, maksimum `30.00` Elo.

Yon olumludur - gerceklesen performansa karsi `+0.001779` Spearman, ve Opta
acigini `-0.065098`den `-0.063319`e daraltir - fakat CI sifiri kesiyor. Iki
seed `0.998326` Spearman ile ortusuyor, dolayisiyla eslesmis testin ayirt
edebilecegi alan zaten dardir: CI genisligi (`~0.013`) etkinin kendisinden
(`~0.0018`) cok buyuktur. Tek sezon ve `236` takimla bu buyuklukte bir etki
olculemez; sonuc katmanin degersiz oldugunu degil, mevcut orneklemin karar
vermeye yetmedigini gosterir.

Not: bu sayi katilim normalizasyonu aktive edilmeden once `+0.002517` idi.
Iki katman kismen ayni kutleye dokundugu icin, katilim duzeltmesi devreye
girince Surprise'in seed ekseninde tasidigi pay kuculdu.

Bu, mac loss ekseniyle ayni sonucu verir (`§6`): isaret pozitif, buyukluk
kucuk, guvenilirlik yok. Katmanin lehine ve aleyhine karar `2026/27`
prospective veriye birakilmistir.

Iki sonuc birlikte modelin yerini kesin olarak konumlandirir. AO First Elo,
tukettigi ham UEFA katsayisinin `+0.154` Spearman uzerindedir ve bu fark
guvenilirdir: statik pipeline karmasikligini hak eder. Ayni zamanda ticari Opta
siralamasinin `-0.063` altindadir ve bu fark da guvenilirdir. Katilim
normalizasyonu ve kupa katkisi aktivasyonlari bu acigi `-0.069`dan
`-0.063`e daraltmistir.

UEFA katsayisi kolu bagimsiz benchmark degildir; `club_points_t_*` girdileri o
katsayinin bilesenleridir. `reference_type=OWN_INPUT` etiketi bir unit testle
sabitlenmistir. Yayinlanmis 2026 katsayisi kullanilmaz, cunku penceresi tahmin
edilen sezonu icerir.

Uc eksen birlikte, olculebilir acigin mac motorunda degil sezon basi seed'inde
oldugunu gosterir.

Kapsam siniri: ClubElo arsivi agirlikla yerlesik kulupleri kapsar ve 2025/26
eslesmeleri `31` gunluk tazelik kapisina takildigi icin eksen 1 `2024/25`'te
biter. Arsivin yeniden cekilmesi benchmark'i guncel sezona tasir.

Ana dosyalar:

```text
reports/external_benchmark/benchmark_report.md
reports/external_benchmark/benchmark_manifest.json
reports/external_benchmark/prediction_uncertainty.csv
reports/external_benchmark/rating_model_summary.csv
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
