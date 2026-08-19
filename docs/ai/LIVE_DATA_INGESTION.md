# Canli Veri Toplama ve Model Calistirma Sozlesmesi

Son guncelleme: **2026-08-18**

Bu belge AO European Elo'nun AkilOyunu.com uzerinde canli calismasi icin gerekli
verileri, kaynak onceligini, cekim zamanlamasini, veritabani modelini ve rating
settlement akisini tanimlar. Kolon seviyesindeki sema icin
[DATA_CONTRACTS.md](DATA_CONTRACTS.md) kullanilmalidir.

## 1. Kapsam ve Mevcut Durum

Repository su anda sunlari icerir:

- AO First Elo hesap motoru,
- AO Live Elo state ve match update motoru,
- input validation ve identity kontrolleri,
- UEFA, FotMob ve TheSportsDB icin batch/research veri builder'lari,
- exact-UTC replay ve pre-match prediction lock arayuzleri,
- missing xG icin goal-margin fallback.

Repository su anda tek basina tam bir production ingestion servisi degildir.
Canliya gecis icin ayrica sunlar kurulmalidir:

- zamanlanmis source fetch worker'lari,
- kalici veritabani,
- provider-to-AO identity eslestirme servisi,
- prediction lock ve settlement queue'su,
- retry/rate-limit ve dead-letter mekanizmasi,
- veri kalite alarmlari,
- AkilOyunu.com read API'si.

Model kodu bu servislerin hesap cekirdegidir; source adapter ve scheduler bu
cekirdegin disinda kalir.

## 2. Iki Ayri Veri Akisi

Canli sistem iki farkli ritimde calisir.

### 2.1 Sezon baslangici akisi

Yilda/sezonda bir kez AO First Elo uretir:

```text
Takim ve kimlik evreni
Ulke katsayilari
Yerel lig/kupa sonucu ve bes sezon lig history
Kulubun bes sezon Avrupa puani ve mac kaniti
                |
                v
            AO First Elo
                |
                v
         Season state initialize
```

### 2.2 Mac bazli canli akis

Her UEFA macinda iki asama vardir:

```text
Fixture -> AO base prediction lock -> ML + Domestic Poisson served prediction
Result + optional xG -> match settlement -> AO Live Elo
Tie tamamlandiysa -> progression event
```

Sezon basi veri sonradan sessizce degistirilmez. Resmi duzeltme gerekiyorsa yeni
snapshot versiyonu uretilir ve etkilenen sezon kontrollu replay edilir.

## 3. Sezon Baslangici Icin Zorunlu Veriler

### 3.1 Kalici kulup kimligi

Her takim icin:

```text
team_id
team_name
country
country_code
domestic_league
```

`team_id`, AO tarafindan yonetilen kalici kimliktir. UEFA, TheSportsDB, FotMob
ve diger provider kimlikleri ayri crosswalk tablosunda tutulur.

Kimlik kurallari:

- Takim adi tek basina primary key olamaz.
- Isim degisikligi veya sponsor adi AO `team_id`yi degistirmez.
- Birlesme, tasinma veya yeni hukuki kulup icin manuel identity karari gerekir.
- Belirsiz eslesme otomatik kabul edilmez.

### 3.2 UEFA ulke/lig gucu

Hedef sezon basinda onceki bes tamamlanmis sezon icin:

```text
points_t_minus_4 ... points_t
```

gerekir. Bu puanlar takim gucu degil, geldigi ulkenin/ligin Avrupa gucu
sinyalidir.

Onerilen kaynak onceligi:

1. UEFA resmi ulke coefficient/ranking verisi.
2. Arsivlenmis ve checksum'i alinmis resmi UEFA ciktilari.
3. Manuel onayli fallback snapshot.

Veri sezon basinda dondurulur. UEFA sonradan resmi duzeltme yaparsa kaynak
snapshot versiyonu artirilir.

### 3.3 Yerel lig ve kupa baglami

Her UEFA katilimcisi icin:

```text
son tamamlanan domestic_position
league_team_count
is_league_champion
is_cup_winner
european_entry_type
onceki bes tamamlanmis sezon position ve team count
```

gerekir.

Kaynak onceligi:

1. Resmi lig/federasyon final tablosu ve kupa sonucu.
2. TheSportsDB Premium schedule/table verisi.
3. Ikinci bagimsiz spor veri kaynagi veya manuel audit.

TheSportsDB otomatik kaynak olarak kullanilsa bile champion/cup ve takim sayisi
sezon bootstrap audit'inde resmi kaynakla karsilastirilmalidir.

Takvim yili ligi kullanan ulkelerde "son tamamlanan sezon" UEFA hedef sezonunun
baslangic cutoff zamaninda tamamlanmis en yeni domestic kampanyadir. Gelecekte
tamamlanacak domestic sezon sonucu geriye donuk seed'e sizdirilmaz.

### 3.4 Kulubun Avrupa gecmisi

Her takim ve bes tamamlanmis sezon icin:

```text
club_points
played
matches
match_cap
```

gerekir.

Kaynak/oneri:

- `club_points`: UEFA resmi kulup sezon puani veya ayni puan sozlesmesiyle
  resmi sonuclardan deterministik hesap.
- `played`: o sezonda en az bir uygun UEFA maci varsa true.
- `matches`: resmi UEFA fixture/result kayitlarindan sayim.
- `match_cap`: model metadata'sidir; ilgili sezon/format politikasiyla
  versioned olarak tanimlanir, provider'dan gelen rastgele alan degildir.

Avrupa'ya ilk kez katilan takim icin history satiri atlanmaz. Bes sezon
`club_points=0`, `played=false`, `matches=0` yazilir; `match_cap` pozitif model
metadata'si olarak kalir.

### 3.5 Static snapshot ciktilari

Her bootstrap su artefact'lari uretmelidir:

```text
source payload hash'leri
identity audit
validation report
normalized dort input tablo
AO First Elo output
initial rating rank
model/config fingerprint
bootstrap timestamp ve cutoff
```

Cutoff zamani, hedef sezon AO First Elo'sunda kullanilabilecek son veri anini
tanımlar.

## 4. Mac Oncesi Zorunlu Fixture Verileri

Her mac icin sonuc bilinmeden once:

```text
match_id
season
kickoff_utc
competition
round
stage
tie_id
home_team_id
away_team_id
is_knockout
is_tie_decider
is_single_match_tie
is_neutral
```

gereklidir.

Kritik metadata:

- `is_single_match_tie`, beraberlik intercept'ini `0.24`ten `0.12`ye
  degistirir. Sonuctan turetilemez. Knockout satirlarda zorunludur; eksik
  gelirse ingestion hata vermeli, sessizce `false` varsaymamalidir.
- `is_neutral`, global ev sahibi avantajini sifirlar.
- `tie_id`, progression'in bir kez uygulanmasini saglar.
- `stage`, KPO ile R16 ve sonrasi stage'lerin karismamasini saglar.
- `is_tie_decider`, progression event'inin hangi mactan sonra olusacagini
  belirler.

Aktif served prediction icin fixture kaydi ayrica frozen Structural Logistic
feature semasini ve en son causal Domestic Poisson state'ini kullanir. Yerel
state sezon basinda `artifacts/production_prediction/` altindaki checkpoint'ten
yuklenir, tamamlanan yerel lig maclariyla exact-UTC sirasinda ilerletilir.
Karsiligi olmayan kulup neutral/no-coverage snapshot alir; sonucu gormeden
once uydurma attack/defence degeri uretilmez.

Prediction servisinin cikisi rating settlement girdisi degildir. Artifact,
feature veya state sorunu halinde final H/D/A Current AO 1X2'ye doner;
`prediction_status`, `fallback_reason`, coverage ve tum artifact hash'leri
kickoff oncesi loglanir.

Structural ML yarisinin calisabilmesi icin fikstur kaydina rolling Avrupa form
feature'lari da eklenmelidir. Bunlar `club_id` bazli, sezonlar arasi uzanan bir
Avrupa mac deposu ister; tek sezonluk girdi kirpik pencere uretir ve satir
sessizce Current AO'ya duser. `scripts/build_2026_27_prediction_features.py`
bu koprunun referans uygulamasidir ve canli servis icin sablondur.

Bu fallback **totaldir**: satir bazli tahmin herhangi bir `Exception` ile
basarisiz olursa o satir Current AO 1X2'ye duser ve batch'in kalani islenmeye
devam eder. Exception tipi `fallback_reason` icine yazilir, dolayisiyla genis
yakalama teshis kaybina yol acmaz. `KeyboardInterrupt` ve `SystemExit`
yakalanmaz; surec durmaya devam eder. Sistematik bir arizanin gorunur olmasi
fallback oraninin alarma baglanmasina dayanir.

## 5. Mac Sonu Zorunlu Sonuc Verileri

```text
home_goals
away_goals
decided_on_penalties
home_penalty_goals
away_penalty_goals
advanced_team_id
result_status
result_basis
```

Skor sozlesmesi:

- Uzatma yoksa 90 dakika field score.
- Uzatma varsa 120 dakika field score.
- Shootout golleri `home_goals/away_goals` icine eklenmez.
- Shootout metadata'si tie kararini aciklar, field sonucu yonunu degistirmez.
- `advanced_team_id`, yalniz tamamlanan tie'da iki takimdan biri olmalidir.

Mac `FINAL/FINISHED` olmadan rating settlement yapilmaz. Abandoned, postponed,
awarded veya teknik sonuc maclari manuel policy gerektirir ve normal result
gibi sessizce islenmez.

## 6. xG Verisi ve Fallback Sozlesmesi

Aktif model iki tarafli uygun xG varsa kullanir:

```text
xg_home
xg_away
xg_analysis_eligible
xg_provider
xg_duration_scope
xg_snapshot_utc
xg_source_match_id
```

Kabul kosullari:

- iki taraf degeri birlikte mevcut,
- degerler finite ve non-negative,
- mac kimligi ve saha skoru eslesiyor,
- shootout xG'ye dahil degil,
- 90/120 field score ile xG zaman kapsami uyumlu,
- provider scope'u audit edilebilir.

Eksik xG sifirla veya ortalamayla doldurulmaz:

```text
xG eligible -> result + goal margin + xG performance
xG missing  -> result + goal margin fallback
```

### 6.1 Mevcut kaynak gercegi

2025/26 batch calismasinda:

```text
FotMob eligible       606 / 961
TheSportsDB raw xG    648 / 961
TheSportsDB eligible  437 / 961
TheSportsDB placeholder supheli 211
```

FotMob coverage UCL ve UEL'de yuksek, UECL'de dusuktur. FotMob public page
endpoint'i authentication istemese de production SLA veya yeniden dagitim
lisansi saglamaz. TheSportsDB xG alaninin provider modeli, penalty ve 90/120
kapsami belgelenmedigi icin FotMob'un otomatik replacement'i degildir.

Production icin uzun vadeli tercih, zaman kapsami ve lisansi belgelenmis tek
bir xG saglayicisidir. Bu saglanana kadar modelin resmi fallback davranisi
korunur.

### 6.2 Onerilen xG settlement zamanlamasi

Bu bolum operasyonel default oneridir; deployment config'inde acikca
dondurulmalidir:

```text
FINISHED sonucu geldi
        |
        v
PENDING_XG, en fazla 90 dakika
        |
        +-- eligible xG geldi -> SETTLED_WITH_XG
        |
        +-- sure doldu        -> SETTLED_GD_FALLBACK
```

Site saha sonucunu hemen gosterebilir; AO Live Elo xG beklerken `PENDING_DATA`
etiketi tasiyabilir. Deadline sonrasi gelen xG sessizce eski ratingi
degistirmez. Resmi reprocess karari verilirse versioned correction/replay
proseduru uygulanir.

## 7. Kaynak Onceligi ve Adapter'lar

| Veri | Primary | Secondary/audit | Production notu |
| --- | --- | --- | --- |
| UEFA fixtures/results | UEFA match feed | TheSportsDB Premium | Primary endpoint icin SLA/terms kontrolu gerekir |
| Stage/tie/advance | UEFA | Kontrollu aggregate derivation | Manual ambiguity queue olmali |
| Country coefficient | UEFA official ranking | Frozen archive | Sezonluk snapshot |
| Club points/history | UEFA official data/result derivation | Audit totals | Ayni puan tanimi korunmali |
| Domestic standings | Resmi lig/federasyon | TheSportsDB Premium | Final coverage audit |
| Cup winner | Resmi federasyon | TheSportsDB | Champion/cup flags manuel kontrol |
| xG | Lisansli provider; gecici FotMob | TheSportsDB comparison | Eksikte GD fallback |
| External ranking | Opta/ClubElo snapshot | UEFA coefficient | Model girdisi degil, validation |

Mevcut batch adapter'lari:

```text
scripts/build_external_elo_benchmark.py
scripts/build_2025_26_xg_dataset.py
scripts/build_thesportsdb_2025_26_dataset.py
scripts/build_domestic_league_dataset.py
```

Bu scriptler canli daemon degildir. Production worker'lari ayni parser ve
identity kurallarini reusable modullere tasiyarak kullanmali; script kodunu
cron icinde kopyalamamalidir.

## 8. Cekim ve Isleme Zamanlamasi

### 8.1 Sezonluk

```text
T-30 gun  takim/turnuva evreni ve identity on kontrolu
T-14 gun  domestic final tablolar ve UEFA history snapshot
T-7 gun   AO First dry run ve validation
T-2 gun   kaynak reconciliation ve manuel identity kapanisi
T-1 gun   AO First freeze, state initialize, publish
```

Buradaki `T`, kapsamdaki ilk UEFA macinin kickoff zamanidir. On eleme dahilse
bootstrap ilk on eleme macindan once tamamlanir.

### 8.2 Fixture polling

Onerilen default:

```text
7+ gun uzakta   gunde 1 kez
1-7 gun uzakta  6 saatte 1
0-24 saat       saatte 1
son 60 dakika   T-60 ve T-15 dogrulama
```

Kickoff degisikligi prediction lock oncesinde guncellenebilir. Lock sonrasi
fixture degisirse prediction iptal/version event'i gerekir.

### 8.3 Pre-match prediction lock

Onerilen default: `kickoff_utc - 10/15 dakika`.

Model lineup kullanmadigi icin lineup beklemek zorunlu degildir. Lock kaydi:

```text
generated_at_utc < kickoff_utc
state/config/model fingerprint
pre-match component ratingleri
H/D/A probabilities
fixture format metadata
```

icerir ve append-only ledger'a yazilir.

### 8.4 Result polling

```text
kickoff sonrasi 90. dakikadan itibaren 2-5 dakikada bir
FINAL olana kadar exponential backoff
FINAL sonrasi ikinci kaynak reconciliation
```

Provider rate limiti ortak token bucket ile korunur. HTTP `429/5xx` retry,
jitter ve max-attempt politikasina tabi olmalidir.

## 9. Ayni Kickoff Batch'i

Ayni UTC aninda baslayan maclar icin:

1. Tek state snapshot al.
2. Tum prediction'lari sonuc olmadan kilitle.
3. Ledger'a yaz.
4. Sonuclari ancak butun prediction'lar kaydedildikten sonra isle.

Bir takim ayni kickoff aninda iki fixture'da bulunamaz. Bu durum high-severity
veri hatasidir.

## 10. Settlement State Machine

Onerilen durumlar:

```text
DISCOVERED
FIXTURE_VALIDATED
PREDICTION_LOCKED
IN_PLAY
PENDING_RESULT_CONFIRMATION
PENDING_XG
SETTLED_WITH_XG
SETTLED_GD_FALLBACK
CORRECTION_REQUIRED
VOID
```

Normal akis:

```text
FIXTURE_VALIDATED
 -> PREDICTION_LOCKED
 -> PENDING_RESULT_CONFIRMATION
 -> PENDING_XG
 -> SETTLED_WITH_XG veya SETTLED_GD_FALLBACK
```

Settlement tek transaction icinde sunlari yazmalidir:

- result observation,
- xG observation/fallback nedeni,
- match update audit,
- iki takimin yeni Power state'i,
- varsa progression event'i,
- yeni AO Live Elo,
- event/config fingerprint.

Qualifier rotasinda competition degisimi state resetlemez. Her fixture'in
canonical `round` degeri Q1/Q2/Q3/Qualifying Play-off/Main eslemesini belirler.
Base stage onemi `0.40/0.55/0.70/0.85`, `%50` qualifier retention ile ayni mac
update'inde `0.20/0.275/0.35/0.425` efektif K'ya donusur. Ilk main fixture
kilitlenmeden once carry, reset veya baska bir mac-disi rating degisimi yapilmaz;
prediction lock son gercek mac sonrasindaki AO Live Elo'yu aynen kullanir.

## 11. Progression Event Akisi

Tie tamamen bittiginde:

1. `tie_id` state'te acik mi kontrol et.
2. `advanced_team_id` iki taraftan biri mi kontrol et.
3. Stage eligible mi kontrol et.
4. Ayni tie daha once bonus almis mi kontrol et.
5. Competition bakiyesi ve cap'i kontrol et.
6. Power settlement'tan sonra winner-only bonusu ekle.

Eligible stage'ler:

```text
ROUND_OF_16
QUARTERFINAL
SEMIFINAL
FINAL
```

Knockout play-off, lig asamasi ve on elemeler bonus uretmez. Penalty ile tur
atlayan takim progression alir; shootout GD/xG sinyali uretmez.

## 12. Onerilen Veritabani Semasi

CSV export desteklenir ancak canli state icin transactional veritabani
onerilir. PostgreSQL uygun bir secimdir.

### `clubs`

```text
team_id PK
canonical_name
country_code
domestic_league
active_from / active_to
```

### `provider_team_identities`

```text
provider + provider_team_id UNIQUE
team_id FK
provider_name
confidence
verified_at
```

### `season_input_snapshots`

```text
snapshot_id PK
target_season
cutoff_utc
source_manifest_sha256
model_version
created_at
```

Normalize static input satirlari `snapshot_id` ile versioned alt tablolarda
tutulur.

### `initial_ratings`

```text
snapshot_id + team_id UNIQUE
ao_first_elo
rank
domestic_prior
european_prior
effective_exposure
domestic_surprise_adjustment
audit_json
```

### `fixtures`

```text
match_id PK
provider_match_id
season, competition, round, stage
kickoff_utc
home_team_id, away_team_id
tie_id
format flags
status
source_version
```

### `prediction_ledger`

```text
prediction_id PK
match_id + model_version + config_id UNIQUE
generated_at_utc
pre-match ratings
expected_home_score
home/draw/away probabilities
fixture fingerprint
state fingerprint
```

### `match_results`

```text
match_id + result_version UNIQUE
field score
penalty metadata
advanced_team_id
confirmed_at
source hashes
```

### `xg_observations`

```text
match_id + provider + snapshot_utc UNIQUE
xg_home, xg_away
duration_scope
analysis_eligible
rejection_reason
source hash
```

### `rating_state`

Her takim icin son state:

```text
season + team_id UNIQUE
ao_first_elo
power_elo
achievement_reserve
progression bonuses
ao_live_elo
last_event_utc
last_match_id
model_version
config_id
row_version
```

Optimistic lock veya transaction row lock ile ayni state'in iki worker
tarafindan cift guncellenmesi engellenir.

### `rating_events`

Append-only event ledger:

```text
event_id PK
match_id UNIQUE, normal settlement icin
pre/post ratings
expected/actual
GD/xG audit
Power Delta
model/config
created_at
```

### `progression_events`

```text
tie_id UNIQUE
competition, stage
recipient_team_id
requested/applied bonus
pre/post competition balance
cap
created_at
```

### `source_payloads` ve `data_quality_audits`

Ham response URI, fetch zamani, HTTP status, checksum, parser version, kalite
kontrolleri ve hata nedenleri saklanir. API key veya secret payload'a yazilmaz.

## 13. Idempotency ve Concurrency

- `match_id` normal settlement icin yalniz bir kez Power event'i uretir.
- `tie_id` yalniz bir kez progression event'i uretir.
- Worker retry ayni idempotency key ile ayni sonucu donmelidir.
- State update transaction sirasinda config/model fingerprint kontrol edilir.
- Chronology gerilemesi reddedilir.
- Ayni kickoff batch tahminleri ortak snapshot ID tasir.
- Settlement ve state publish atomik olmalidir.

## 14. Resmi Duzeltme ve Replay

Settled mac skoru resmi olarak degisirse eski state uzerine ikinci Delta
eklenmez. Prosedur:

1. Eski result versiyonunu superseded olarak isaretle.
2. Yeni resmi payload ve nedeni kaydet.
3. Etkilenen mactan onceki guvenilir state checkpoint'ini yukle.
4. O mac ve sonraki event'leri exact-UTC sirayla replay et.
5. Yeni state versiyonunu atomik publish et.
6. Degisen public rating ve prediction audit'ini raporla.

Bu yol nadir ve kontrollu olmalidir. Sessiz in-place edit replay zincirini
bozar.

## 15. AkilOyunu.com Read API Ciktilari

Onerilen public read yuzeyi:

### Takim ratingi

```text
team_id
team_name
season
ao_first_elo
ao_live_elo
rank
last_updated_utc
last_match_id
model_version
data_status
```

Kullaniciya tek ana puan `ao_live_elo` gosterilir. Power ve progression
bilesenleri aciklama/audit endpoint'inde bulunabilir.

### Mac tahmini

```text
match_id
generated_at_utc
home/draw/away probabilities
home/away pre-match AO Live Elo
format ve neutral metadata
model_version
prediction_status=LOCKED
```

### Mac update audit

```text
field result
expected score
Power Delta
goal multiplier
xG applied/fallback
progression bonus
home/away post-match AO Live Elo
settled_at_utc
```

## 16. Monitoring ve Alarm Esikleri

### Kaynak sagligi

- fixture feed son basarili fetch yasi,
- HTTP hata/429 orani,
- source payload parse failure,
- provider schema degisikligi,
- xG coverage ve gecikme,
- official-secondary score mismatch.

### Veri kalitesi

- unmapped team count > 0,
- duplicate match/tie,
- missing kickoff veya non-UTC,
- ayni takim/ayni kickoff collision,
- invalid field/penalty score,
- ambiguous stage/tie,
- xG single-sided veya scope mismatch.

### Model invariantlari

- olasilik toplami `1 +/- 1e-12`,
- match Power zero-sum hatasi `<=1e-9`,
- progression cap ihlali `0`,
- KPO progression event'i `0`,
- chronology regression `0`,
- model/config mismatch `0`.

### Operasyonel SLA onerisi

```text
fixture freshness         < 60 dakika, match day
prediction lock success   > 99.5%
result settlement         FINAL sonrasi < 15 dakika, xG beklemiyorsa
xG settlement             FINAL sonrasi <= 90 dakika
unmapped production team  0
duplicate settlement      0
```

## 17. Guvenlik, Lisans ve Secret Yonetimi

- `THESPORTSDB_API_KEY` environment/secret manager'da tutulur.
- `.env.local`, raw credential ve auth header Git'e girmez.
- Public endpoint'in erisilebilir olmasi yeniden dagitim lisansi anlamina
  gelmez.
- FotMob veya diger page-data source production'a alinmadan once terms,
  ticari kullanim ve cache/republication kosullari incelenir.
- Provider raw payload erisimi internal tutulur; public API yalniz izin verilen
  derived degerleri yayimlar.
- Secret rotation ve rate-limit kullanimi deployment runbook'unda kayitlidir.

## 18. Launch Oncesi Kabul Kriterleri

- [ ] Tum sezon katilimcilari kalici AO `team_id` ile eslesiyor.
- [ ] Static dort input tablo validation'dan geciyor.
- [ ] AO First output ve baslangic state'i freeze edildi.
- [ ] Fixture primary/secondary reconciliation calisiyor.
- [ ] Tek mac, neutral, stage, tie ve decider metadata'si dogru.
- [ ] Prediction sonucu bilinmeden once ledger'a kilitleniyor.
- [ ] Result field score ve shootout ayrimi test edildi.
- [ ] xG grace/fallback politikasi deployment config'inde donduruldu.
- [ ] Match ve progression idempotency testleri gecti.
- [ ] Power zero-sum ve probability invariant alarmlari aktif.
- [ ] Resmi correction replay proseduru test edildi.
- [ ] Read API cache invalidation ve atomic state publish test edildi.
- [ ] Source lisans/terms kontrolu tamamlandi.
- [ ] API key ve secret'lar repository disinda.
- [ ] 2026/27 prospective prediction ledger production baslamadan hazir.

## 19. Uygulama Onceligi

Canli entegrasyon icin onerilen teknik sira:

1. PostgreSQL semasi ve migration'lar.
2. Kalici identity/provider crosswalk servisi.
3. UEFA fixture/result adapter'i ve raw payload store.
4. Season bootstrap worker'i ve AO First freeze.
5. Prediction lock scheduler'i.
6. Result settlement worker'i, once GD fallback ile.
7. xG adapter'i ve 90 dakikalik grace state'i.
8. Progression reconciliation.
9. AkilOyunu.com read API'si.
10. Monitoring, correction replay ve prospective evaluation dashboard'u.

Bu sira, xG kaynagi tamamen hazir olmasa bile guvenli bicimde core modelin
canliya alinmasini saglar. Eksik xG maclari contract geregi GD fallback ile
calisir; veri sonradan iyilestikce xG coverage artar.
