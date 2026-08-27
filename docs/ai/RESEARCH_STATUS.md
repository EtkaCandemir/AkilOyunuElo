# Aktif, Shadow ve Research Durumu

Bu belge repository'deki cok sayidaki deneyin production modeliyle
karistirilmamasini saglar. Son guncelleme: **2026-08-27**.

## European Exposure Production Karari

*(Asagidaki sayilar `2026-08-27` katilim normalizasyonu aktivasyonundan
once olculmustur; guncel model metrikleri `reports/current_model/`
altindadir.)*

AO First Elo effective European exposure tavani `0.85`ten `0.65`e indirildi.
Sabit `benchmark=20`, prior scale `1.00`, competition quality `1/1/1` ile
`4.884` unseen macta Brier `0.569287`, log-loss `0.960299` ve accuracy
`0.554259` elde edildi. Brier ve log-loss `6/6` foldda iyilesti; UCL, UEL ve
UECL segmentlerinin tamami olumlu kaldi. Spearman ve pairwise nokta tahminleri
az geriledi fakat sezon-block guven araligi guvenilir zarar gostermedi.

Salt loss optimumu `0.60` olsa da ranking-first sabit aday `0.65` secildi.
`benchmark=24 + exposure=0.65` kombinasyonu guvenilir Spearman zarari nedeniyle
production'a alinmadi. Kanit:
`output/european_exposure_cap_backtest_2018_2026/`.

## 1. Durum Tanimlari

| Durum | Anlam |
| --- | --- |
| `ACTIVE` | Production contract ve active config tarafindan kullaniliyor |
| `PROMOTE_CANDIDATE` | Backtest kapilarini gecti; ayri aktivasyon/onay bekliyor |
| `KEEP_SHADOW` | Canli ratingi degistirmeden izlenebilir; kanit production icin yetersiz |
| `KEEP_DIAGNOSTIC` | Hipotezi anlamak icin yararli, urun adayi degil |
| `DISABLED` | Contract yuzeyinde olabilir ama etkisi sifir |
| `REJECTED` | Testte zarar, kararsizlik veya yapisal sorun nedeniyle kullanilmaz |

Bir output JSON'da `PROMOTE_CANDIDATE` yazmasi production aktivasyonu degildir.
`production_activated` ve ana contract ayrica kontrol edilir.

## 2. Aktif Production Katmanlari

| Katman | Durum | Aktif davranis |
| --- | --- | --- |
| AO First v2 | `ACTIVE` | 500-2000 referans olcegi; no clipping |
| Country strength | `ACTIVE` | benchmark 25, gamma .80, hard upper tail beta 0 |
| Domestic achievement | `ACTIVE` | position, champion, weighted cup contribution `w=0.129032`, unknown finish at the `0.15` floor |
| Club European history | `ACTIVE` | benchmark 20, beta 0 |
| European Exposure | `ACTIVE` | .60 season + .40 match; effective cap .65 |
| European prior katilim normalizasyonu | `ACTIVE` | `rate = history x (1+k) / (pw + k)`, `k = 0.20`; tam katilimda notr |
| Domestic Surprise | `ACTIVE` | theta .40, variance .50, cap +/-30, 5 seasons |
| Dynamic Power Elo | `ACTIVE` | Scale 835.561, H 148.544, K 103.981, season carry 0 |
| Continuous qualifier retention | `ACTIVE` | Base `0.40/0.55/0.70/0.85`, retention `0.50`, effective `0.20/0.275/0.35/0.425`, MAIN reset yok |
| Score-preserving 1X2 | `ACTIVE` | draw .24, shape 1 |
| Single-match draw | `ACTIVE` | format metadata ile draw .12 |
| Controlled GD | `ACTIVE` | alpha .15, tau 300, cap 4 |
| Bounded xG | `ACTIVE` | ratio .30, scale 1.25; missing -> GD fallback |
| Fixed progression | `ACTIVE` | R16/QF/SF/FINAL, 12/8/4, cap 48/32/16 |
| ML + Domestic Poisson 1X2 | `ACTIVE_WITH_MONITORING` | `%50/%50` log blend, `rho=0`, AO fallback, rating feedback yok |

## 3. En Guclu Prediction-Only Challenger'lar

Bu modeller AO First/Live ratingi degistirmez; yalniz pre-match 1X2
olasiliklarini iyilestirmeyi hedefler.

### 3.1 Domestic Dynamic Poisson + AO blend araştırması

Durum: Full-history `rho=0.15` adayı **`PROMOTE_CANDIDATE`** olarak kalır;
bu aday production'da aktif değildir. Production ensemble bunun yerine daha
basit `AO_POISSON_RHO0_CONTROL` kolunu kullanır.

Veri:

```text
45,423 yerel lig maci
19 lig
508 source takim
Tarihsel backtest cekirdegi 171 AO kulubunu guvenle eslestirir. Canli
checkpoint UCL/UEL kapsama genislemesiyle 312 dogrulanmis kimligi denetler;
runtime kalite kapisindan gecen 311 kimligi kullanir. 2026/27 hedef evreninde
79/80 kulup yeterli yerel kanita sahiptir. União Torreense iki sezon/40 mac
esigini gecmedigi icin takim-bazli Poisson profili kapali; ilgili maclarda
`ONE/NONE` coverage kuraliyla guvenli davranis uygulanir.
4,884 unseen Avrupa maci
```

Pooled sonuc:

```text
AO baseline Brier     0.572093
AO+Poisson Brier      0.569044   delta -0.003049
AO baseline log-loss  0.964371
AO+Poisson log-loss   0.960298   delta -0.004073
Fold wins             5/6 Brier, 5/6 log-loss
```

Secilen full-history aday:

```text
team learning rate 0.02
season carry 0.90
shrinkage 10
team venue context false
attack coefficient 0.09210
defence coefficient 0.06066
rho 0.15
AO/Poisson blend weight 0.50
```

Dosya:
`output/domestic_poisson_backtest_2018_2026/selected_candidate.json`.

Aktif production bileşeninde aynı domestic state parametreleri korunur fakat
`rho=0` kullanılır. Bu ayrım contract ve artifact manifestinde dondurulmuştur.

### 3.2 Structural ML blend

Standalone tarihsel gate durumu: **`KEEP_SHADOW`**. Dondurulmuş Structural
Logistic modeli, manuel operasyon kararıyla monitored final ensemble içinde
`%90` ağırlıklı Current ML bileşeni olarak aktiftir; tek başına servis edilmez.

```text
Brier     0.568690   AO'ya fark -0.003402
Log-loss  0.960458   AO'ya fark -0.003913
Fold      4/6 Brier, 3/6 log-loss
```

Pooled loss güçlü olsa da standalone fold ve calibration kapıları tamamlanmadığı
için bağımsız production modeli değildir.

### 3.3 ML + Poisson final ensemble

Tarihsel otomatik gate: **`KEEP_SHADOW`**. Guncel operasyonel karar:
**`PROMOTE_WITH_MONITORING`**. Bu katman production tahmin servisinde aktiftir,
fakat AO First/Live rating state'ine geri beslenmez.

```text
Brier     0.562152   AO'ya fark -0.004360
Log-loss  0.950022   AO'ya fark -0.006371
Accuracy  0.561220   AO'ya fark +0.002662
```

Bu degerler `2026-08-27` katilim normalizasyonu aktivasyonundan sonra
yeniden uretilmistir.

Current ML'ye karsi `4/6` Brier ve `4/6` log-loss fold kazanmistir. Pooled ve
kalibrasyon sonuclari iyidir; ancak dependency uncertainty gate gecmedigi ve
bazi sezon/coverage segmentleri hafif ters yone gittigi icin otomatik terfi
almamistir.

2026/27 icin dondurulan production tahmini:

```text
ML weight      0.50
Poisson weight 0.50
Poisson source AO_POISSON_RHO0_CONTROL
rating feedback false
```

Dosya:
`output/final_prediction_ensemble_backtest_2018_2026/selected_candidate.json`.

Runtime artifact otoritesi:
`artifacts/production_prediction/manifest.json`. Artifact veya feature sorunu
oldugunda satir Current AO 1X2'ye duser; her fallback loglanir.

## 4. Diger Shadow/Diagnostic Katmanlar

| Katman | Durum | Temel gerekce |
| --- | --- | --- |
| Team venue context | `KEEP_SHADOW_CANDIDATE` | 5/6 fold ve pooled iyi; dependency CI sifiri kesiyor |
| Draw shape 0.84 | `KEEP_SHADOW` | Brier 4/6, log 3/6; competition no-harm gecmedi |
| Genellestirilmis kupa katkisi | `ACTIVE` | `2026-08-27`de aktive edildi (`w=0.129032`); tasarim karari, envelope sifiri kesiyor |
| Forecast Scale ~898-906 | `KEEP_SHADOW` | Pooled loss ve fold kapilari gecmedi |
| COVID/non-COVID H | `KEEP_SHADOW_PENDING_METADATA` | Closed-door match metadata yok; sezon proxy guvenli degil |
| Domestic Surprise MOB | `KEEP_DIAGNOSTIC` | Pooled loss iyi; tekrar eden anlamli split yok |
| Opponent quintile | `KEEP_DIAGNOSTIC` | Persistence Spearman 0.015, cap-hit %36, segment harm |
| Opponent tercile | `KEEP_DIAGNOSTIC/REJECT` | Pooled loss kotulesti, persistence 0.003 |
| Scoreline Poisson/DC | Diagnostic | Skor/O-U/BTTS icin yararli; rating feedback yok |
| xG bilgili gol beklentisi (Elo-only taban) | `KEEP_SHADOW` | xG, gol formunu 4/4 fold gecer; yalniz `XG_PRESENT` segmentinde guvenilir |
| Exposure cap x katilim etkilesimi | `KEEP_CURRENT_CAP` | `0.40`-`0.85` boyunca hicbir cap `0.65`ten guvenilir farkli degil; her CI sifiri kesiyor |
| European prior katilim normalizasyonu | `ACTIVE` | `2026-08-27`de aktive edildi. Seed Spearman `+0.009434`, kor kontrole karsi `+0.011925`, Brier `-0.002051`; ucu de guvenilir, dokuz kapi gecti |
| Domestic Surprise guclendirme | `REJECTED` | Pooled kazanc sifirdan ayirt edilemiyor; sezon ici ranking guvenilir bozuluyor; foldlar hicbir adayda anlasmiyor |
| xG form terimi domestic attack/defence uzerinde | `KEEP_SHADOW_CANDIDATE` | Reponun en iyi skor kolunu ve gol kontrolunu `PHASE:MAIN` ve `XG_PRESENT`'te guvenilir gecer; pooled envelope sifiri kesiyor |
| Format P-advance | Diagnostic | Tie probability; rating state'i degistirmez |
| New-format calibration | Diagnostic | Yalniz iki sezon, ayri production fit icin az |
| AO First seed asimetrisi boost | `KEEP_DIAGNOSTIC` | Domestic-form kolu 6 foldun yalniz birinde ve tek takimda etkili; Opta acigini kapatmadi |

### 4.1 AO First seed asimetrisi boost

Ince fakat negatif Avrupa gecmisi olan bir takimin, sifir Avrupa gecmisli bir
takimdan daha fazla asagi cekilmesi hipotezi `1.887` takim-sezon ve `4.884`
unseen Avrupa macinda test edildi. `BOOST_BLIND` yalniz kuralin mekanik
kontroludur ve production adayi olamaz. Bagimsiz domestic-form sinyali, her
sezonun ilk UEFA macindan once dondurulan causal Domestic Poisson durumundan
uretilmistir.

```text
                         Brier farki   Log-loss farki   Seed Spearman farki
BOOST_BLIND              -0.000448     -0.000675        +0.000509
BOOST_DOMESTIC_FORM      -0.000065     -0.000122        +0.000293
```

Blind kontrol `186/1413` test takim-sezonunu etkiledi fakat 2025/26 Opta
Spearman acigini `-0.06955`ten `-0.07272`ye buyuttu. Domestic-form kolu yalniz
`1/1413` takim-sezonu etkiledi; 2025/26'da hicbir takimi degistirmedigi icin
Opta acigi baseline ile ayni kaldi. Ranking conservative envelope sifiri
kesti; guvenilir ranking iyilesmesi yoktur. Karar `KEEP_DIAGNOSTIC` ve
production aktivasyonu `false`tir.

`BOOST_LONG_HISTORY`, repository'de tarihli 10-20 yillik Opta/UEFA snapshot'i
veya denetlenmis uzun donem basari tablosu olmadigi icin
`UNAVAILABLE_NOT_RUN` olarak kaydedildi. Ayrintili paket:
`reports/ao_first_seed_boost/`.

## 5. Kapali veya Reddedilmis Rating Katmanlari

### Dynamic K

Exposure, match count ve inactivity ile takim bazli K degisimi production
kapilarini gecmedi. Aktif K sabittir.

### Competition K

Normal match learning rate'e UCL/UEL/UECL carpanlari uygulanmaz. Rakip gucu
zaten expected score icindedir; sert turnuva K oranlari loss/ranking faydasi
vermedi.

### Qualification-stage K

Q1'den qualifying play-off'a dogru artan yedi K profili, butun turlari `1.00`
kullanan production referansina karsi `2018/19-2025/26` doneminde test edildi.
Nested kol pooled Brier ve log-loss'u kucuk miktarda iyilestirdi ve qualifier
giris hareketini azaltti; ancak same-season ranking tum foldlarda geriledi ve
forward ranking yalniz `3/5` geciste iyilesti. Karar `KEEP_CURRENT_K`; profil
production'a alinmadi. Ayrintili yerel rapor
`output/qualification_stage_k_backtest_2018_2026/backtest_report.md` altindadir.

### Qualification-stage K + main-stage carry

Stage-K ile qualifier kazaniminin ana asamaya tasinma oranini birlikte test eden
ikinci calisma, `FULL/MILD/BALANCED` stage profilleri ile `0.50/0.60/0.75/1.00`
carry adaylarini karsilastirdi. Ana asama maclari ve forward ranking karar
hedefi olarak kullanildi; qualifier same-season ranking veto olmadi.

Nested aday qualifier girisindeki P95 mutlak hareketi `203.79` Elo'dan
`109.14` Elo'ya indirdi. Buna karsilik pooled ana-asama Brier farki
`+0.000079`, log-loss farki `+0.000200` oldu; UCL iyilesirken UECL geriledi.
Fold secimi Brier'da `4/6`, log-loss'ta `3/6`, forward ranking'de `4/5`
kazandi. Karar `KEEP_REFERENCE` oldu ve production degismedi. Full-history
loss-safe denge adayi `FULL_CARRY_060` olarak kaydedildi; bu sonuc prospective
aktivasyon kaniti degildir. Daha sonra asama onemi ve urun guvenligi gerekcesiyle
explicit urun karari verildi: Q1/Q2/Q3/QPO/Main `0.40/0.55/0.70/0.85/1.00`
ve qualifier-to-main gap carry `0.50` gecici production sozlesmesine aktive
edildi. Bu gecici davranis daha sonra asagidaki continuous-retention
sozlesmesiyle supersede edildi. Ayrintili rapor
`output/qualification_stage_k_carry_backtest_2018_2026/backtest_report.md`
altindadir.

Sonraki urun incelemesinde tek seferlik MAIN-entry carry'nin takim tur gectigi
anda mac olmadan gorunen Elo'yu dusurebildigi belirlendi. Kullaniciya gosterilen
ve tahminde kullanilan tek Power state'ini korumak icin `%50` retention her
qualifier macinin stage K degerine gomuldu. Aktif efektif carpanlar
`0.20/0.275/0.35/0.425`; MAIN `1.00` ve geciste reset yoktur. Bu aktivasyon
acik urun karari olup onceki nested backtest sonucunun yeniden yorumlanmasi
degildir; 2026/27 monitoring kayitlarinda ayrica izlenecektir.

### Season Power Carry

Onceki sezon sonu Power'i yeni AO First ile karistirma production'da kapali.
`power_carry=0`.

### Achievement Reserve

Reserve base sifirdir ve aktif contract'ta disabled'dir. Kodda geriye uyumluluk
icin yuzey bulunabilir; runtime ratinge katkisi yoktur.

### Zero-sum progression

Turu gecene bonus, elenenden ayni bonusu dusme yaklasimi reddedilmistir.
Production winner-only fixed progression kullanir.

### KPO progression

Knockout play-off rotasi yeni formatta seribasi takimlara karsi asimetrik ve
gucle negatif korelasyonlu oldugu icin eligible set'ten cikarilmistir.

### Stage-weighted progression

Cesitli gentle/linear/final-heavy cap gridleri ranking-first kapilari gecmedi.
Production fixed four-stage 12/8/4 sozlesmesini korur. KPO'yu veya bes stage'i
referans alan eski research ciktilari current contract icin stale kabul edilir.

### Opponent profile rating feedback

Q1-Q5/Q1-Q3 rakip profilleri kalici takim ozelligi olarak sonraki sezonda
tekrarlanmadi. Prediction-only diagnostic disinda ratinge eklenmez.

### Red-card ve match-specific post-event feature'lar

Pre-match tahminde bilinmeyen veya maliyetli/kararsiz veri oldugu icin active
rating formulu icinde yoktur.

## 6. Manuel Aktivasyonlar

Su katmanlar tamamen otomatik istatistik kapisindan degil, backtest + model
mantigi + acik urun karariyla aktiftir:

- Domestic Surprise,
- controlled goal margin,
- bounded xG,
- fixed progression.

Bu, katmanlarin testsiz oldugu anlamina gelmez. Ancak prospective kanit
yorumunda manuel karar olmasi acikca belirtilmelidir.

### Genellestirilmis yerel kupa katkisi

Aktif model lig ve kupa basarisini `max` ile birlestirir, bu da kupayi katki
degil **taban** yapar. Lig skoru kupa tabaninin (`0.62`) uzerinde olan her kupa
sahibi kupasindan sifir kredi alir; `2025/26` orneginde `51` kupa sahibinin
`35`'i (`%69`) bu durumdadir. Duble bonusu ayrica yalniz sampiyon-ve-kupa
ciftinde calisir.

Test edilen tek parametreli genelleme:

```text
Achievement = min(cap, max(L, C) + weight * min(L, C))
```

Kupa kazanmayanda `min(L, C) = 0` oldugu icin katman tanim geregi inerttir.
`weight = 0.129032` mevcut sampiyon-ve-kupa bonus buyuklugunu korur ve ayni
mantigi her lig+kupa kombinasyonuna genisletir.

`2018/19-2025/26` walk-forward sonucu:

```text
Yapisal kapilar            4/4 PASS
Etkilenen kutle            %19.2 takim-sezon (363 kupa sahibi)
Brier fold wins            6/6  (w=0.05 ve w=0.08)
Log-loss fold wins         6/6  (w=0.05 ve w=0.08)
Pooled Brier delta         -0.000118 (w=0.08)
Conservative envelope      guvenilir iyilesme YOK, guvenilir zarar YOK
Rating ekseni              tum CI'lar sifiri kesiyor
Nested secim               6 foldun 5'i w=0
```

Karar `KEEP_SHADOW` idi ve **`2026-08-27` tarihinde aktive edilmistir**
(`w = 0.129032`, `champion_equivalent_weight`). Aktivasyon istatistiksel terfi
**degildir**: nested secim agirligi `0`'a ceker, yani veri kupa katkisi
*eklemeyi* degil mevcut duble bonusu *kaldirmayi* tercih eder. Gerekce kuralin
yapisal olarak yanlis olmasidir - `2026/27`de `53` kupa sampiyonunun `39`'u
kupasindan sifir kredi aliyordu ve etki ters yonluydu: ligde ne kadar kotuysen
kupa o kadar degerliydi. Ayni kanit sinifi Domestic Surprise ve progression ile
aynidir.

Secilen agirlik onceki kuralin zaten odullendirdigi grubu birebir korur:
sampiyon+kupa toplami iki kuralda da `1.0800`. Aktivasyon sonrasi ablation
maliyeti isaret degistirdi: `-0.000080` -> `+0.000022`.

Mantiksal bosluk gercekti fakat kapatilmasi sekiz
sezonda guvenilir bir iyilesme uretmez. Nested secimin `w=0`'a cekmesi ayrica
bagimsiz bir gozlem tasir: mevcut sampiyon-ve-kupa duble bonusunun da olculebilir
destegi yoktur.

Kanit profili, contract'ta hali hazirda **aktif** olan kontrollu gol farki
katmaniyla ayni sinifta olduğu icin bu bir tutarlilik sorusudur; iki katman da
otomatik kapiyi gecmez, biri manuel urun karariyla aciktir.

Rapor: `reports/cup_achievement/` ve
`output/cup_achievement_backtest_2018_2026/backtest_report.md`.

Bu sorunun kalici olarak izlenmesi icin `run_current_model_evaluation.py` artik
`ABLATION_NO_CUP_DOUBLE_BONUS` kolunu tasir. Kol bir olcum aracidir; production
duble bonusu acik kalir. Her degerlendirme kosusunda su tablo uretilir
(kaynak: `reports/current_model/model_summary.csv`, `4884` unseen mac):

```text
ABLATION_NO_SINGLE_MATCH_DRAW   brier maliyeti  +0.000714
ABLATION_NO_XG                  brier maliyeti  +0.000674
ABLATION_NO_DOMESTIC_SURPRISE   brier maliyeti  +0.000132
ABLATION_NO_GOAL_MARGIN         brier maliyeti  +0.000061
ABLATION_NO_CUP_DOUBLE_BONUS    brier maliyeti  +0.000022
ABLATION_NO_PROGRESSION         brier maliyeti  -0.0000003
```

Pozitif maliyet katmanin faydali oldugunu gosterir.

**Bu tablo, referans ve no-surprise kollarinin da `0.65` exposure cap'iyle
seed'lendigi duzeltmeden sonra uretilmistir.** Duzeltme oncesinde bu iki kol
bayat bir static manifest uzerinden `0.85` cap kullaniyordu; production kolu
ise yeniden uretilmis `0.65` artifact'ini okuyordu. `1887` takim-sezonun
`949`u ortalama `23.12` (maks `156.49`) Elo farkliydi ve bu fark tamamen
Domestic Surprise ablation'ina yaziliyordu.

Bu yuzden daha once raporlanan `+0.000476` -> `+0.002726` sicramasi katmanin
buyumesi degil, referans kolunun cap kazancini kaybetmesiydi: fark `+0.00225`,
exposure cap kararinin kendi olculmus kazanciyla (`-0.00225` Brier) birebir
ayni. Duzeltilmis deger `+0.000287`, cap oncesi olculen `+0.000476` ile ayni
mertebededir. `0.85` referansiyla uretilmis butun eski kopyalar stale kabul
edilmelidir.

Ayni pakette her ablation kolunun conservative envelope'i da tasinir. **Iki**
katmanin kapatilmasi **guvenilir zarar** verir:

```text
ABLATION_NO_SINGLE_MATCH_DRAW   +0.000714   CI [+0.000003, +0.001688]   reliable harm
ABLATION_NO_XG                  +0.000674   CI [+0.000036, +0.001369]   reliable harm
ABLATION_NO_DOMESTIC_SURPRISE   +0.000132   CI [-0.000189, +0.000401]   degil
ABLATION_NO_GOAL_MARGIN         +0.000061   CI [-0.000126, +0.000234]   degil
ABLATION_NO_CUP_DOUBLE_BONUS    +0.000022   CI [-0.000202, +0.000217]   degil
ABLATION_NO_PROGRESSION         -0.0000003  CI [-0.000006, +0.000005]   degil
```

**Tek mac beraberlik katmaninin `reliable harm` isareti orneklem gucuyle
degil yapisal gerekceyle savunulur.** Katmanin `224` tek macinin `200`'u
(`%89`) `2020/21` COVID sezonundan gelir, COVID sonrasi segment `24` mactir
ve Brier CI'sinin alt siniri `+0.000003`. Ayni pakette log-loss ekseni zaten
sifiri kesiyor. Bu katmani `2827` uygun maca dayanan bounded xG ile ayni
guven sinifinda sunmak yaniltici olur.

Domestic Surprise artik guvenilir zarar veren katman degildir. `ACTIVE`
kalmasinin gerekcesi degismedi - aktivasyon zaten `PROMOTE_MANUAL` idi ve
seed/ranking ekseninde olculmustu - fakat "en degerli katman" iddiasi
geri cekilmistir.

`2026-08-27` katilim normalizasyonu aktivasyonundan sonra maliyet `+0.000287`
-> `+0.000133`e daha da inmistir. Iki katman kismen ayni kutleye - Avrupa
kaniti ince kulupler - dokundugu icin katilim duzeltmesi Surprise'in tasidigi
isin bir kismini ustlenir. Bu bir zarar degil, is bolumu degisikligidir.

Bu tablonun bir siniri vardi: katmanin gerekcesi mac tahmini degil sezon basi
seed kalitesidir, ve o eksende dis referansa karsi surprise-off kolu yoktu. Kol
`run_current_external_benchmark.py` icine `MODEL_ABLATION` olarak eklendi ve
kosuldu (`2025/26`, `236` kulup):

```text
AO First Elo                    Spearman 0.421435   pairwise 0.646117
AO First Elo, Surprise kapali   Spearman 0.420773   pairwise 0.646045

AO - AO(Surprise kapali)     +0.000662   CI [-0.006582, +0.007451]   guvenilir degil
AO - Opta                    -0.065211   CI [-0.107079, -0.022522]
AO(Surprise kapali) - Opta   -0.065873   CI [-0.108393, -0.022593]
```

Kolun ilk kosusunda (katilim normalizasyonu aktive edilmeden once) ayni fark
`+0.002517` idi. Iki katman kismen ayni kutleye dokundugu icin katilim
duzeltmesi devreye girince Surprise'in seed ekseninde tasidigi pay kuculdu.

Katman `236` kulubun `181`ini oynatir (ortalama `8.90`, maks `30.00` Elo), yonu
olumludur ve Opta acigini `-0.071275`ten `-0.068758`e daraltir; fakat CI sifiri
kesiyor. Iki seed `0.998612` Spearman ile ortustugu icin eslesmis testin ayirt
edebilecegi alan dardir: CI genisligi (`~0.006`) etkiden (`~0.0025`) buyuktur.
Tek sezon ve `236` kulup bu buyuklukte bir etkiyi cozemez.

Uc eksen de ayni seyi soyluyor: isaret pozitif, buyukluk kucuk, guvenilirlik
yok. Katmani buyutme sorusu ayrica test edilip `REJECTED` almisti (asagi bkz.),
dolayisiyla acik olan tek yon bekleyip `2026/27` prospective veriye bakmaktir.

Progression'in maliyeti yedinci ondalikta sifirdir ve
CI'si simetrik olarak sifiri sarar; kapatmak da acmak da olculebilir bir sey
degistirmez. Kupa duble bonusu ise maliyeti **anlamli olcude negatif** olan tek
katmandir: kapatmak modeli kucuk miktarda iyilestirir. Fark guvenilir degildir,
bu yuzden karar `KEEP` olarak kalir ve `2026/27` prospective verisiyle
kendiliginden netlesecektir.

### Bounded xG katmani: alti sezonluk yeniden dogrulama

Aktif `xg_performance` katmani tek sezonluk kanitla acilmisti (`961` mac,
`606` uygun, `manual_product_decision`). O karar aninda yalniz `2025/26` xG'si
vardi. FotMob xG `2020/21`'e kadar uzandigi icin veri seti
`data/xg_2020_2026/` altinda `4884` mac ve `2827` uygun xG'ye genisletildi.

`2018/19` ve `2019/20` disarida kalir: FotMob bu sezonlar icin xG alanini hic
dondurmez. Sinir sondajla dogrulanmistir.

```text
XG_SIX_SEASON     Brier 0.573732   log-loss 0.966444
XG_2025_26_ONLY   Brier 0.574160   log-loss 0.967043
NO_XG             Brier 0.574346   log-loss 0.967315
```

Conservative envelope alti sezonluk kolda **her segmentte** guvenilir iyilesme
verir:

```text
ALL          -0.000614   CI [-0.001149, -0.000108]   reliable
PHASE:MAIN   -0.001169   CI [-0.002103, -0.000210]   reliable
XG_PRESENT   -0.001385   CI [-0.002465, -0.000278]   reliable
```

Kazanc xG'nin bulundugu yerde yogunlasir; bulunmadigi yerde etki `+0.000006`,
yani katman bilgi yokken gurultu eklemez. Kapsam ana asamada `%98.7`, on
elemede `%11`'dir.

Bu yeni bir katman degildir. Kosu yalniz mevcut kararin kanit tabanini `4.7`
kat genisletir; asil fark modelden degil veriden gelir.

Acik urun onayiyla contract'in `xg_performance_evidence` alani guncellenmistir
(`production_revision: 2026-08-19-xg-evidence-revalidated-six-seasons`).
Katman parametreleri **degismemistir**: `ratio 0.30`, `scale 1.25`,
`floor 0.70`. Ilk aktivasyon kaydi `superseded_evidence` altinda korunur.
`tests/test_xg_multiseason.py` hem parametrelerin sabitligini hem kanit
sayilarinin veri setiyle tutarliligini pinler.

Rapor: `reports/xg_multiseason/` ve
`output/xg_multiseason_backtest_2020_2026/backtest_report.md`.

Paylasilan `XG_DATA` sabiti bu calismayla birlikte alti sezonluk haritaya
yonlendirilmistir. O anda canonical evaluation'da `CURRENT_PRODUCTION` Brier
degeri `0.572093` -> `0.571537` inmis ve katman deger tablosunda xG ucuncu
siradan birinci siraya cikmisti (`+0.000797`).

Bu siralama daha sonra iki kez guncellenmistir. `2026-08-21` exposure cap
yayilimindan sonra Brier `0.569287`e indi ve xG maliyeti `+0.000759` oldu; o
anda birinci sira gorunuste Domestic Surprise'a gecti (`+0.002726`). Ancak o
sicrama bayat `0.85` referans seed'inden geliyordu. Referans kolu duzeltilince
xG **birinci sirada kalir** ve tek guvenilir katman olur; ayrinti ve duzeltilmis
tablo `§6`'dadir.

Onemli tekrar-uretilebilirlik notu: `domestic_surprise_*`, `opponent_quintile`
ve benzeri arastirma scriptlerinin daha once uretilmis ciktilari tek sezonluk
xG haritasiyla hesaplanmistir. Yeniden kosuldugunda farkli deger uretirler.
Bu bir hata degildir - dogru veriyle hesaplanacaklardir - ancak dondurulmus
kanitla karsilastirirken bilinmelidir.

### xG bilgili gol beklentisi

Aktif skor katmani her iki orani yalniz Elo farkindan turetir; takim formu
yoktur. Bu calisma her iki tarafa birer form terimi ekler ve ayni yapiyi iki
kaynakla besler. `GOALS` kolu kontroldur.

```text
XG        exact_score_nll 3.027008   Elo'ya fark -0.002834
GOALS     exact_score_nll 3.029642   Elo'ya fark -0.000201
ELO_ONLY  exact_score_nll 3.029843
```

Conservative envelope yalniz bir segmentte guvenilir iyilesme verir:

```text
XG / XG_PRESENT / exact_score_nll   -0.006970   CI [-0.011187, -0.002668]
```

Gol kontrolunde hicbir segment guvenilir degildir, yani kazanc xG'ye ozgudur.
xG'nin bulunmadigi maclarda kol hafifce zarar verir (`+0.003065`), bu yuzden
katman kosullu olmalidir.

**Kritik sinir:** bu kosunun tabani (`ELO_ONLY`) reponun mevcut
`DOMESTIC_ATTACK_DEFENCE_POISSON` kolundan daha basittir ve o kol daha iyi bir
`exact_score_nll` (`3.016594`) tutturur. Gosterilen sey "xG basit bir tabana
eklendiginde yardim eder"dir. Bu sinir asagidaki takip calismasiyla
kaldirilmistir.

Karar `KEEP_SHADOW`. Rapor: `reports/xg_goal_expectation/`.

### xG form terimi domestic attack/defence uzerinde

Yukaridaki kosunun kendi yazdigi sinir buydu: kazanc basit bir tabanda
gosterilmisti. Bu calisma ayni form terimini reponun en iyi skor kolunun
**uzerine** koyar, yerine degil. Uc kol, ayni `3528` mac, ayni dort fold
(`2022/23`-`2025/26`), ayni dort metrik:

```text
DOMESTIC_AD                 3.021498   taban, reponun en iyi skor kolu
DOMESTIC_AD_GOALS_FORM      3.021068   -0.000430   kontrol
DOMESTIC_AD_XG_FORM         3.019349   -0.002149   aday
```

Taban zayiflatilmis bir kopya degildir: yayinlanmis
`DOMESTIC_ATTACK_DEFENCE_POISSON` ayni `3528` macta `3.021437` tutturur, fark
`+0.000061` nats. `tests/test_xg_domestic_goal_model.py` `none` kolunun
`fit_european_poisson_transfer` ile ayni katsayilari urettigini pinler.

Kazancin ne kadari hayatta kaldi:

```text
xG formu, ELO_ONLY tabani uzerinde    -0.002834
xG formu, DOMESTIC_AD tabani uzerinde -0.002149    %76 hayatta
XG_PRESENT segmentinde                -0.006970 -> -0.005104    %73 hayatta
```

Domestic attack/defence ile xG formu buyuk olcude ayni bilgiyi tasimaz;
onceki kazancin domestic modelin zaten bildigi seyin tekrari oldugu hipotezi
elenmistir.

Conservative envelope, tabana karsi:

```text
XG_FORM   / PHASE:MAIN  -0.006106   CI [-0.009441, -0.002732]   reliable
XG_FORM   / XG_PRESENT  -0.005104   CI [-0.008347, -0.001876]   reliable
XG_FORM   / ALL         -0.002149   CI [-0.005292, +0.000502]   degil
GOALS_FORM / hicbiri                                            degil
```

Iki kol da ayni iki ek parametreyi tasidigi icin aralarindaki dogrudan
karsilastirma kazanci kaynaga baglar:

```text
XG_FORM vs GOALS_FORM / PHASE:MAIN  -0.004385   CI [-0.007279, -0.001362]   reliable
XG_FORM vs GOALS_FORM / XG_PRESENT  -0.003796   CI [-0.006653, -0.001064]   reliable
XG_FORM vs GOALS_FORM / ALL         -0.001719   CI [-0.004352, +0.000659]   degil
```

Egitim penceresi duyarliligi ayni sonucu verir. Yalniz xG kapsanan sezonlarla
egitildiginde xG kolu gol kontrolunu `4/4` foldda gecer ve gol kontrolu tabana
zarar verir (`+0.000354`).

Duran sinirlar:

- Pooled `ALL` guvenilir degildir. Ornegin `%41`'inde xG yoktur ve orada kol
  hafifce zarar verir (`+0.002066`), yani katman kosullu olmalidir.
- Dogru kosullandirma kurali **test edilmemistir**. `XG_ABSENT` zarari mevcut
  macin xG'sinden gelmez - form terimi yalniz gecmisi okur - bu maclarin
  cogunlugu on eleme turudur ve oradaki takimlarin xG **gecmisi** incedir.
  Kapi "mac xG tasiyor mu" degil "takimin xG gecmisi yeterli mi" olmalidir.
- Turev pazarlar hala gecmez: ust 2.5 Brier `-0.000213`, CI
  `[-0.001023, +0.000491]`. Skor katmanini `Diagnostic`'e koyan asil gerekce
  degismemistir.

Karar `KEEP_SHADOW_CANDIDATE`. Production contract'a dokunulmamis, aktivasyon
yapilmamistir. Rapor: `reports/xg_domestic_goal_expectation/` ve
`output/xg_domestic_goal_expectation_backtest_2020_2026/backtest_report.md`.

### Domestic Surprise guclendirme

Aktif Domestic Surprise'in katsayilarini buyutmenin - `theta`, domestic cap,
exposure ailesi - sezon basi seed'ini iyilestirip iyilestirmedigi test edildi.
`206` aday, alti fold, nested secim.

```text
Pooled Brier farki      -0.000011   CI [-0.000357, +0.000408]   guvenilir degil
Pooled log-loss farki   -0.000032   CI [-0.000504, +0.000520]   guvenilir degil
Fold kazanimi           Brier 4/6, log-loss 4/6
```

**Kazanc, exposure cap aktivasyonu tarafindan zaten alinmisti.** Calisma iki
kez kosuldu; ilki tabanini eski contract'tan (`0.85` cap), ikincisi guncel
contract'tan (`0.65`) aldi:

```text
eski contract tabani     -0.000555
guncel contract tabani   -0.000011      53 kat kucullme
```

Karsilastirma icin cap degisikliginin kendi kazanci unseen pencerede `-0.00225`
Brier ve `6/6` fold; amplification onun ustune `-0.000011` ekliyor.

Iki kapi kaliyor:

```text
same_season_spearman           -0.002021   CI [-0.003998, -0.000530]   ZARAR
same_season_pairwise_accuracy  -0.001329   CI [-0.002183, -0.000492]   ZARAR
fold secim modal payi           0.333      kapi 0.50                   BASARISIZ
```

Alti foldda bes farkli config secildi: `theta` `0.4`-`1.75`, domestic cap
`30`-`150`, variance penalty `0`-`0.75`, uc exposure ailesi. `206` adaylik bir
yuzey her zaman fold basina bir kazanan uretir; kazananlar birbirini tutmuyorsa
pooled fark aramanin ozelligidir, gonderilebilir bir parametre setinin degil.

Etki, Avrupa'da en az oynayan kuluplerde yogunlasiyor (`0-.20` exposure
bandinda ortalama `25.22` Elo, `.40-.65` bandinda `8.95`) - yani tahmin
kalitesine en az katki yapan ama public ranking'de en gorunur olan yer.

Karar `REJECTED`. Rapor: `reports/domestic_surprise_amplification/`.

#### Duzeltilen uc olcum hatasi

Yukaridaki sonuc, calismanin ilk halindeki uc hatanin duzeltilmesinden sonra
uretilmistir. Ucu de ayni yone isaret ediyordu:

1. **Ranking veto'su loss sign convention'ini kullaniyordu.** `reliable_harm`
   `lower > 0` ile hesaplaniyordu; ranking skorlarinda yuksek iyi oldugu icin
   dogru test `upper < 0`. Iki guvenilir zarar `False` raporlaniyordu.
2. **Fold secim kararliligi hic olculmuyordu.** Artik
   `fold_selection_modal_share_at_least_half` kapisi var ve her iki terfi
   kademesini de veto ediyor.
3. **Kontrol-artifact kontrolu yapisi geregi basarisiz oluyordu.** Dondurulmus
   artifact onceki exposure cap ile yazilmisti; cap dusunce cap'e degen her
   kulubun domestic agirligi `1-0.85` -> `1-0.65` degisir. Kontrol ikiye
   ayrildi: cap altinda rebuild artifact'i birebir uretmeli (maks fark
   `5.4e-13`), cap ustunde her fark cap'te olmali (maks `6.000000`, tam olarak
   `(0.35-0.15) x 30`). Bug degil, migration.

Duzeltmeden once karar `AMPLIFY_WITH_MONITORING`'di, sonra `KEEP_CURRENT`.
Stored-baseline kolu da `KEEP_CURRENT` veriyordu fakat yanlis calisan kontrol
check'i uzerinden - dogru cevaba kazara variyordu.

### European prior katilim normalizasyonu

European Prior sabit agirlikli bir toplamdan gelir ve **girilmeyen sezon sifir
puan katkisi yapar** (`features.py:122`, `validators.py:337`). Girilip hic puan
alinamayan sezon da sifir katki yapar, dolayisiyla prior su ikisini ayirt
edemez: "oynadim, kotuydum" ve "katilamadim". Ikincisi domestic prior'in zaten
sahiplendigi bir olgudur ve iki kez faturalandirilir - bir kez dusuk history,
bir kez dusuk exposure agirligi.

Bu kosu yalniz birinci faturayi kaldirir; exposure agirligina hic dokunmaz:

```text
rate = weighted_european_history × (1 + k) / (weighted_season_exposure + k)
```

`pw = 1` iken `rate` yayinlanmis history'nin aynisidir, yani bes sezonluk tam
kaniti olan `644` kulup **tanim geregi** hic hareket etmez. Etkilenen kutle
`1013` kulup-sezon (`%53.7`); hic girmemis `230` kulup exposure `0` tasidigi
icin seed'i degismez.

Uc kol, alti fold, `1413` kulup-sezon:

```text
BASELINE                   Spearman 0.459768
PARTICIPATION_BLIND_LIFT            0.457303   -0.002465   <- kontrol ZARAR veriyor
PARTICIPATION_NORMALIZED            0.469202   +0.009434
```

Kor kontrol ayni kutleyi ayni ortalama buyuklukte, yalniz exposure'in
fonksiyonu olarak yukari iter. Zarar vermesi kazancin **katilim yapisindan**
geldigini gosterir.

```text
seed Spearman, tabana karsi     +0.009462   CI [+0.004316, +0.015733]   guvenilir
seed Spearman, kontrole karsi   +0.011925   CI [+0.006652, +0.017587]   guvenilir
Brier                           -0.002051   CI [-0.002964, -0.001181]   guvenilir
log-loss                        -0.002904   CI [-0.004186, -0.001704]   guvenilir
```

Aday hem tabani hem kontrolu `6/6` foldda gecer. Mac loss ekseni no-harm
kontrolu olarak tasinmisti; zarar vermemekle kalmayip iyilestirdi. Olcek icin
exposure cap aktivasyonunun kendi kazanci `-0.00225` Brier idi; bu katman onun
**ustune** `-0.00205` ekler.

Opta ekseni (tek sezon, `236` kulup, diagnostik): taban `0.417957`, aday
`0.421107`; acik `-0.068661` -> `-0.065511`.

Duran sinirlar: kutle buyuk (`%53.7`, p95 seed hareketi `97.4` Elo, kapi `100`);
`k` tam kararli degil (modal pay `0.667`, foldlar `0.75/0.20/0.20/0.20/0.20/0.00`);
servis edilen ML/Poisson katmani yeniden secilmedi.

Karar `PROMOTE_CANDIDATE` idi ve **`2026-08-27` tarihinde aktive edilmistir**
(`production_revision: 2026-08-27-participation-cup-and-unknown-position`,
`k = 0.20`). Aktivasyon oncesinde iki ek kapi kosuldu ve ikisi de gecti:

- **Exposure cap etkilesimi** (`reports/participation_exposure_interaction/`):
  `0.40`-`0.85` araligindaki hicbir cap `0.65`ten guvenilir farkli degil, ve
  replay edilen dort cap icinde en iyi loss aday egride de `0.65`te. Cap
  degismedi.
- **Servis edilen katman** (`reports/participation_served_ensemble/`): tam
  zincir (feature store -> ML -> Poisson -> ensemble) iki kolda bastan
  kuruldu. Servis edilen katmanin kendi AO tabanina kattigi deger
  `-0.004450` -> `-0.004360`; hicbir segmentte `reliable_harm` yok.

Rapor: `reports/european_participation/`.

### Exposure cap x katilim normalizasyonu etkilesimi

Katilim normalizasyonu, cap'in korumak icin var oldugu guvenilmezligin bir
kismini onarir. Bu, cap'in ayarini yeniden acar: `0.65`i secen tarama
dokunulmamis prior uzerinde kosmustu.

Ayni `14` degerlik grid iki kez tarandi - production prior'i ve
katilim-normalize prior uzerinde. `k` her cap'te her fold'un icinde yeniden
secildi, yani iki egri kendi en iyi ayarinda karsilastirildi.

**Iki katman birbirinin yerine gecmiyor.** Katilim kazanci cap boyunca
`+0.007260` … `+0.009953` araliginda, neredeyse sabit.

**Cap'i yukseltmenin kazanci katilim tarafindan yutuluyor:**

```text
cap 0.65 -> 0.70 ranking kazanci
  taban egrisi        +0.000564
  katilim egrisi      -0.000015
```

**Ve hicbir cap `0.65`ten guvenilir sekilde farkli degil.** Sezon-blogu
bootstrap, `seed_spearman`, alti fold:

```text
0.600   -0.000728   CI [-0.003213, +0.001246]   degil
0.700   -0.000016   CI [-0.000778, +0.000680]   degil
0.750   +0.001091   CI [-0.000673, +0.002854]   degil
0.775   +0.001574   CI [-0.000250, +0.003646]   degil   <- egrinin tepesi
0.850   +0.001238   CI [-0.001344, +0.004139]   degil
```

Guvenilir iyi olan cap yok, guvenilir kotu olan da yok. Egriden tepe okumak
bir tercihtir, olcum degildir.

Olcum ayirt edemedigi icin karar baska gerekcelerle verildi ve ucu de `0.65`i
isaret ediyor: replay edilen dort cap icinde en iyi Brier `0.65`te
(`0.568884`; `0.70` `+0.000092`, `0.775` `+0.000418`); `k` kararliligi `0.65`te
en yuksek (modal pay `0.67`, `0.70` ve `0.775`te `0.33`); ve `0.65` aktive
edilmis degerdir, degistirmek icin guvenilir bir fark gerekir.

Olcek: tum cap boyutu `+0.0016` degerinde ve guvenilir degil; katilim
duzeltmesi `+0.0094` ve guvenilir.

Karar `KEEP_CURRENT_CAP`. **Katilim katmani aktive edilirse cap sorusu tekrar
sorulmalidir** ve bu kosu yeniden uretilmelidir. Rapor:
`reports/participation_exposure_interaction/`.

### Exposure cap aktivasyonunun seed zincirine yayilmasi

`2026-08-21` exposure cap kararindan sonra contract `0.65` ilan ediyordu fakat
servis edilen seed hala `0.85` ile uretilmis dondurulmus artifact'ten
geliyordu. Zincir uc seviyeydi:

```text
domestic_surprise_features.csv           effective_european_exposure maks 0.85
  -> selected_candidate_team_adjustments.csv
    -> canonical evaluation, model_end_ratings, dis benchmark
```

Kok yeniden uretildi ve zincir yukari kosuldu. `949 / 1887` kulup-sezon
degisti (cap'e degenler, ortalama `45.97` Elo); cap altindaki `938` kulup tam
olarak sifir degisti. Sonuc birebir tuttu: canonical evaluation artik
contract'in ilan ettigi `brier 0.569287`, `log_loss 0.960299`,
`accuracy 0.554259` degerlerini uretiyor. Sezon ici Spearman `+0.007994`,
dis benchmark AO First Elo `0.416620` -> `0.417957`.

Yayilim iki yan etki gosterdi ve ikisi de kayda gecirilmistir:

1. Domestic Surprise'in ablation maliyeti gorunurde `+0.000476` -> `+0.002726`
   cikti ve katman **en degerli** konuma gecti. **Bu okuma yanlisti ve geri
   cekilmistir.** Zincirin bir halkasi yayilimin disinda kalmisti: canonical
   evaluation'in referans ve no-surprise kollari seed'lerini
   `output/v2_dynamic_calibration_2018_2026/selected_dynamic_model.json`
   icindeki `static_config`ten aliyordu ve orasi hala `0.85` ilan ediyordu.
   Boylece ablation "surprise kapali" ile "cap 0.85'e geri dondu"yu birlikte
   olcuyordu. Manifest `0.65`e cekilip evaluation yeniden kosuldugunda
   `CURRENT_PRODUCTION` degismedi, referans `0.573699` -> `0.571199` iyilesti
   ve surprise maliyeti `+0.000287` (CI sifiri kesiyor) oldu.
   `run_current_model_evaluation.py` artik manifest cap'i contract cap'ine
   esit degilse kosuyu durdurur.
2. Gamma sensitivity yuzeyi variance penalty olarak `0.5` yerine `0.7`
   secmeye basladi. Contract `0.5` ilan ettigi icin aday `0.5`'e sabitlendi ve
   yuzeyin tercihi bastirilmak yerine raporlandi. **Variance penalty
   yuzeyinin tercihi exposure cap'ine bagimlidir**; bunu degistirmek ayri bir
   karar ve ayri kanit ister.

## 7. Siradaki Kanit Onceligi

1. 2026/27 lig asamasi ve sonrasi locked prospective AO prediction ledger.
2. Ayni ledger uzerinde AO, Domestic Poisson, Structural ML ve 50/50 ensemble.
3. xG icin cok sezonlu tek saglayici ve tutarli 90/120 scope.
4. Team venue challenger icin pre-match attendance/closed-door metadata.
5. Progression katmaninin ranking katkisi yoksa kapatma urun karari.
5b. Domestic Surprise'in seed ekseninde tek sezonluk `+0.002517` Spearman
   yonu; `236` kulup ayirt etmeye yetmiyor. Ayni ablation kolunu birden cok
   sezonluk dis snapshot uzerinde kosmak tek gercek cozumdur.
6. ~~xG form terimini `DOMESTIC_ATTACK_DEFENCE_POISSON` uzerine ekleyen kol.~~
   **Yapildi (2026-08-19).** Sonuc olumlu: form terimi reponun en iyi skor
   kolunun uzerinde de kazandirir ve gol kontrolunu `PHASE:MAIN` ile
   `XG_PRESENT` segmentlerinde guvenilir gecer. Skor katmanini
   `Diagnostic`'ten cikarmaya tek basina yetmez; asagidaki iki madde kalan
   engellerdir.
7. xG **gecmis** esigine gore kosullandirilmis form kolu. Kazanc xG'nin
   bulundugu yerde yogunlasir, bulunmadigi yerde kol hafifce zarar verir; dogru
   kapi macin xG tasiyip tasimadigi degil, takimin xG gecmisinin yeterli olup
   olmadigidir. `XG_ABSENT` zarari yapisal olarak kapanirsa pooled envelope da
   kapanabilir.
8. Skor katmaninin turev pazarlari. Ust 2.5 ve KG var hala climatology
   sinirindadir ve skor katmanini `Diagnostic`'te tutan asil gerekce budur;
   skor dagilimindaki iyilesme bu iki urune henuz gecmemistir.

## 8. Research Sonucu Production'a Alma Protokolu

1. Candidate artifact ve feature schema fingerprint'ini dondur.
2. Rating feedback'in false oldugunu test et veya rating modeli ise yeni
   conservation sozlesmesini acikla.
3. Prospective ledger'da baseline ile ayni mac evrenini kullan.
4. Pooled + fold + competition + missing-data segmentlerini raporla.
5. CI, calibration ve operational fallback'i incele.
6. Kullanici explicit activation karari verdikten sonra contract/config/API
   degisikligini ayri commit olarak yap.
