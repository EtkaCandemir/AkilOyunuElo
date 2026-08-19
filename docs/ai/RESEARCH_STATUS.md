# Aktif, Shadow ve Research Durumu

Bu belge repository'deki cok sayidaki deneyin production modeliyle
karistirilmamasini saglar. Son guncelleme: **2026-08-18**.

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
| Domestic achievement | `ACTIVE` | position, champion, cup, true-double |
| Club European history | `ACTIVE` | benchmark 20, beta 0 |
| European Exposure | `ACTIVE` | .60 season + .40 match; effective cap .85 |
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
171 AO'ya guvenle eslesen kulup
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
Brier     0.568093   AO'ya fark -0.003999
Log-loss  0.959242   AO'ya fark -0.005129
Accuracy  0.553849   AO'ya fark +0.003686
```

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
| Genellestirilmis kupa katkisi | `KEEP_SHADOW` | `w=0.05/0.08` icin 6/6 fold; conservative envelope sifiri kesiyor |
| Forecast Scale ~898-906 | `KEEP_SHADOW` | Pooled loss ve fold kapilari gecmedi |
| COVID/non-COVID H | `KEEP_SHADOW_PENDING_METADATA` | Closed-door match metadata yok; sezon proxy guvenli degil |
| Domestic Surprise MOB | `KEEP_DIAGNOSTIC` | Pooled loss iyi; tekrar eden anlamli split yok |
| Opponent quintile | `KEEP_DIAGNOSTIC` | Persistence Spearman 0.015, cap-hit %36, segment harm |
| Opponent tercile | `KEEP_DIAGNOSTIC/REJECT` | Pooled loss kotulesti, persistence 0.003 |
| Scoreline Poisson/DC | Diagnostic | Skor/O-U/BTTS icin yararli; rating feedback yok |
| xG bilgili gol beklentisi | `KEEP_SHADOW` | xG, gol formunu 4/4 fold gecer; yalniz `XG_PRESENT` segmentinde guvenilir |
| Format P-advance | Diagnostic | Tie probability; rating state'i degistirmez |
| New-format calibration | Diagnostic | Yalniz iki sezon, ayri production fit icin az |

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

Karar: **`KEEP_SHADOW`**. Mantiksal boşluk gercektir fakat kapatilmasi sekiz
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
duble bonusu acik kalir. Her degerlendirme kosusunda su satir uretilir:

```text
ABLATION_NO_SINGLE_MATCH_DRAW   brier maliyeti  +0.000658
ABLATION_NO_DOMESTIC_SURPRISE   brier maliyeti  +0.000491
ABLATION_NO_XG                  brier maliyeti  +0.000242
ABLATION_NO_GOAL_MARGIN         brier maliyeti  +0.000194
ABLATION_NO_PROGRESSION         brier maliyeti  -0.00000007
ABLATION_NO_CUP_DOUBLE_BONUS    brier maliyeti  -0.000057
```

Pozitif maliyet katmanin faydali oldugunu gosterir. Kupa duble bonusu, aktif
katmanlar icinde maliyeti **negatif** olan tek katmandir: kapatmak modeli kucuk
miktarda iyilestirir. Fark conservative envelope'da guvenilir degildir
(`CI [-0.000123, +0.000019]`), bu yuzden karar `KEEP` olarak kalir ve
`2026/27` prospective verisiyle kendiliginden netlesecektir.

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
yonlendirilmistir. Canonical evaluation'da `CURRENT_PRODUCTION` Brier degeri
`0.572093` -> `0.571537` iner ve katman deger tablosunda xG ucuncu siradan
**birinci** siraya cikar (`+0.000797`).

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
eklendiginde yardim eder"dir. Siradaki test xG form terimini domestic
attack/defence modelinin **uzerine** koymaktir.

Karar `KEEP_SHADOW`. Rapor: `reports/xg_goal_expectation/`.

## 7. Siradaki Kanit Onceligi

1. 2026/27 lig asamasi ve sonrasi locked prospective AO prediction ledger.
2. Ayni ledger uzerinde AO, Domestic Poisson, Structural ML ve 50/50 ensemble.
3. xG icin cok sezonlu tek saglayici ve tutarli 90/120 scope.
4. Team venue challenger icin pre-match attendance/closed-door metadata.
5. Progression katmaninin ranking katkisi yoksa kapatma urun karari.
6. xG form terimini `DOMESTIC_ATTACK_DEFENCE_POISSON` uzerine ekleyen kol;
   skor katmanini `Diagnostic`'ten cikarmak icin gereken kritik test.

## 8. Research Sonucu Production'a Alma Protokolu

1. Candidate artifact ve feature schema fingerprint'ini dondur.
2. Rating feedback'in false oldugunu test et veya rating modeli ise yeni
   conservation sozlesmesini acikla.
3. Prospective ledger'da baseline ile ayni mac evrenini kullan.
4. Pooled + fold + competition + missing-data segmentlerini raporla.
5. CI, calibration ve operational fallback'i incele.
6. Kullanici explicit activation karari verdikten sonra contract/config/API
   degisikligini ayri commit olarak yap.
