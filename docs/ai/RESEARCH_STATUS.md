# Aktif, Shadow ve Research Durumu

Bu belge repository'deki cok sayidaki deneyin production modeliyle
karistirilmamasini saglar. Son guncelleme: **2026-08-13**.

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
| Dynamic Power Elo | `ACTIVE` | Scale 835.561, H 148.544, K 103.981, carry 0 |
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
| Forecast Scale ~898-906 | `KEEP_SHADOW` | Pooled loss ve fold kapilari gecmedi |
| COVID/non-COVID H | `KEEP_SHADOW_PENDING_METADATA` | Closed-door match metadata yok; sezon proxy guvenli degil |
| Domestic Surprise MOB | `KEEP_DIAGNOSTIC` | Pooled loss iyi; tekrar eden anlamli split yok |
| Opponent quintile | `KEEP_DIAGNOSTIC` | Persistence Spearman 0.015, cap-hit %36, segment harm |
| Opponent tercile | `KEEP_DIAGNOSTIC/REJECT` | Pooled loss kotulesti, persistence 0.003 |
| Scoreline Poisson/DC | Diagnostic | Skor/O-U/BTTS icin yararli; rating feedback yok |
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

## 7. Siradaki Kanit Onceligi

1. 2026/27 lig asamasi ve sonrasi locked prospective AO prediction ledger.
2. Ayni ledger uzerinde AO, Domestic Poisson, Structural ML ve 50/50 ensemble.
3. xG icin cok sezonlu tek saglayici ve tutarli 90/120 scope.
4. Team venue challenger icin pre-match attendance/closed-door metadata.
5. Progression katmaninin ranking katkisi yoksa kapatma urun karari.

## 8. Research Sonucu Production'a Alma Protokolu

1. Candidate artifact ve feature schema fingerprint'ini dondur.
2. Rating feedback'in false oldugunu test et veya rating modeli ise yeni
   conservation sozlesmesini acikla.
3. Prospective ledger'da baseline ile ayni mac evrenini kullan.
4. Pooled + fold + competition + missing-data segmentlerini raporla.
5. CI, calibration ve operational fallback'i incele.
6. Kullanici explicit activation karari verdikten sonra contract/config/API
   degisikligini ayri commit olarak yap.
