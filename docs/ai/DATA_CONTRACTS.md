# Veri Sozlesmeleri

Bu belge production static ve dynamic veri giris/cikislarini tanimlar. Kaynak
sema `validators.py`, `pipeline.py` ve `dynamic_csv.py` dosyalaridir.

## 1. Genel Kurallar

- Kimlik kolonlari bos olamaz ve string olarak kararlidir.
- Sayilar finite olmalidir; `NaN`, `inf`, `-inf` ve negatif degerler ilgili
  alan izin vermedikce reddedilir.
- Duplicate anahtarlar sessizce birlestirilmez.
- `season`, rating'in uretildigi hedef sezon veya macin ait oldugu sezondur.
- UTC timestamp timezone-aware olmali ve chronology `kickoff_utc + match_id`
  ile deterministik tutulmalidir.
  CSV ve prediction girisi eksik timezone'a UTC atamaz; acik offset'i UTC'ye
  donusturur. Domestic gol alanlari tam sayi olmalidir: `1.0` kabul,
  `0.999999` ve `1.000001` ret; toleransla kabul edip asagi kesme yoktur.
- `played_*`, ML `is_single_match_tie`, `is_neutral`, `is_knockout` icin
  canonical `true/false` metni boolean/0/1 ile ayni sonucu verir.
- ML baseline ve metadata'da ortak tur/leg kolonlari esitse korunur, eksik
  taraf digerinden tamamlanir; dolu ve celisen degerler reddedilir.
- Eksik kanit sifirla doldurulmaz. Yalniz model sozlesmesinin acik sifir kabul
  ettigi satirlar sifir olabilir.

## 2. Static Input: `teams.csv`

Anahtar:

```text
team_id
```

Zorunlu kolonlar:

| Kolon | Anlam |
| --- | --- |
| `team_id` | AO kalici kulup kimligi |
| `team_name` | Gorunen takim adi |
| `country` | Ulke adi |
| `country_code` | Kararli ulke kodu |
| `domestic_league` | Yerel lig adi |

Bir `team_id` yalniz bir kez bulunur. Saglayici kimlikleri bu tablodaki kalici
AO kimliginin yerine kullanilmaz.

## 3. Static Input: `country_coefficients.csv`

Anahtar:

```text
season + country_code
```

Zorunlu kolonlar:

```text
season
country
country_code
points_t_minus_4
points_t_minus_3
points_t_minus_2
points_t_minus_1
points_t
```

Tum `points_*` alanlari non-negative finite sayidir. `t`, hedef sezondan once
tamamlanan en yeni sezondur.

Opsiyonel audit alanlari:

```text
official_five_year_total
official_country_rank
```

Audit alanlari formulde kullanilmaz. Varsa sayisal olarak validate edilir.

## 4. Static Input: `domestic_context.csv`

Anahtar:

```text
season + team_id
```

Dosya tek bir hedef sezon icermelidir.

Zorunlu kolonlar:

| Kolon | Sozlesme |
| --- | --- |
| `season` | Hedef sezon |
| `team_id` | AO kalici kimligi |
| `domestic_position` | Bos veya `1..league_team_count` tam sayi |
| `league_team_count` | Pozisyon varsa `>=2` tam sayi |
| `is_league_champion` | Gecerli boolean |
| `is_cup_winner` | Gecerli boolean |
| `european_entry_type` | Entry metadata; bos olamaz |

Domestic Surprise icin opsiyonel ama cift halinde verilmesi gereken kolonlar:

```text
history_position_t_minus_5 ... history_position_t_minus_1
history_team_count_t_minus_5 ... history_team_count_t_minus_1
```

Her sezon icin position ve team count ya birlikte dolu ya birlikte bos
olmalidir. Bes tam sezon yoksa Surprise adjustment sifir olur.

Semantik kurallar:

- Champion ve bilinen position celisemez; champion icin position `1`dir.
- Bilinmeyen position champion flag'ini gecersiz kilmaz.
- Bir takimlik lig gecersizdir.
- Kupa kazanmak tek basina double bonus uretmez.

### 4.1 Sezon bootstrap kaynak kaniti

`domestic_context.csv` hesap girdisidir; kaynak kaniti ayrica
`domestic_history_audit.csv` dosyasinda tutulur. Kupa sampiyonu rotalari icin
filtrelenmis `cw_domestic_evidence_audit.csv` ayni satirlarin kolay denetlenen
alt kumesidir. Audit en az su bilgileri korur:

```text
route
current_position_vintage
current_position_source
current_source_url
current_source_cache
current_table_position
qualification_route_position
position_crosscheck
current_table_selection_method
current_standings_candidate_count
history_window
```

Hedef sezon `2026/27` ise current vintage `2026` olur: sonbahar-ilkbahar
liglerinde tamamlanan `2025/26`, takvim yili liglerinde cutoff'ta tamamlanmis
`2025` kampanyasi kullanilir. `CW` rotasi icin eski sezon fallback'i gecersizdir.
Guncel ust ligde bulunamayan takimda `domestic_position` bos kalir; eski veya
tahmini pozisyon yazilmaz.

Kaynak sayfasinda birden cok siralama tablosu varsa yalniz siralari benzersiz ve
sirali bicimde tam `1..N` olan tablo adaylari kabul edilir. En buyuk aday ana ust
lig tablosudur; boylece "position by round" veya kismi play-off tablosu current
position kaynagi olamaz.

## 5. Static Input: `club_european_points.csv`

Anahtar:

```text
season + team_id + country_code
```

Zorunlu base kolonlar:

```text
season
team_id
team_name_source
country_code
```

Her `k in {t_minus_4,t_minus_3,t_minus_2,t_minus_1,t}` icin:

```text
club_points_k
played_k
matches_k
match_cap_k
```

Kurallar:

- `played_k` tanimli boolean'dir.
- `matches_k` non-negative tam sayidir.
- `club_points_k` non-negative finite sayidir.
- `match_cap_k` her zaman pozitif finite sayidir.
- `played=false` ise matches ve points `0` olmalidir.
- `played=true` ise matches en az `1` olmalidir.
- Avrupa gecmisi olmayan takim icin satir atlanmaz; bes sezon points/played/
  matches degerleri acik sifir yazilir ve cap'ler pozitif kalir.

Opsiyonel audit alanlari:

```text
official_club_coefficient
country_part
```

Bu alanlar AO First formulu icinde kullanilmaz.

## 6. Static Output: AO First Elo

`pipeline.OUTPUT_COLUMNS` output kolonlarinin otoritesidir. Kolon gruplari:

### Kimlik ve metadata

```text
season, model_version, team_id, team_name, country, country_code,
domestic_league, competition, entry_round, domestic_position,
league_team_count
```

### Country/league strength audit

```text
weighted_country_score
country_strength_uncapped_norm
league_strength_norm
league_strength
country_strength_saturated
country_tail_excess
country_tail_active
```

### Domestic achievement audit

```text
domestic_position_percentile
league_finish_score
cup_base_score
cup_contribution_enabled
cup_contribution_weight
cup_double_bonus
domestic_achievement_uncapped_score
domestic_achievement_score
achievement_saturated
achievement_cap_active
domestic_prior
```

### Domestic Surprise audit

```text
domestic_surprise_enabled
domestic_surprise_status
domestic_surprise_current_finish_score
domestic_surprise_history_seasons
domestic_surprise_historical_mean
domestic_surprise_historical_variance
domestic_surprise_historical_volatility
domestic_surprise_normalized_volatility
domestic_surprise_consistency_multiplier
domestic_surprise_raw_score
domestic_surprise_effective_score
domestic_surprise_domestic_adjustment
adjusted_domestic_prior
```

### European history/exposure audit

```text
weighted_european_history
european_participation_enabled
european_participation_shrinkage
european_history_rate
european_history_uncapped_norm
european_history_norm
european_prior
european_history_saturated
european_tail_excess
european_tail_active
weighted_season_exposure
weighted_match_exposure
european_exposure
effective_european_exposure
exposure_saturated
exposure_tail_excess
exposure_tail_active
saturation_count
```

### Nihai rating

```text
ao_first_elo_before_domestic_surprise
domestic_surprise_ao_first_elo_adjustment
ao_first_elo
ao_first_elo_rank
rating_source_type
validation_warnings
```

## 7. Dynamic Fixture Input

Sonucsuz pre-match prediction girdisi:

```text
match_id
season
kickoff_utc
competition
round
tie_id
is_knockout
is_tie_decider
is_single_match_tie
stage
home_team_id
away_team_id
is_neutral
```

`is_single_match_tie` fixture formatidir. Sonuctan sonra turetilmesi leakage
olur. `tie_id`, knockout state ve tek uygulama progression icin kararlidir.

Bu kolon `is_knockout=true` satirlarda **zorunludur** ve `dynamic_csv` tarafindan
zorlanir. Eksik veya bos deger hata verir; varsayilana dusulmez. Sebep: eksik
bayrak ile acik `false` ayirt edilemez, dolayisiyla bir varsayilan butun tek
maclik tie'lari (finaller dahil) sessizce iki ayakli beraberlik intercept'iyle
fiyatlar. Lig/grup asamasi satirlarinda kolon opsiyoneldir.

## 8. Dynamic Match Input

Settlement girdisi fixture kolonlarina ek olarak:

```text
home_goals
away_goals
xg_home
xg_away
xg_analysis_eligible
decided_on_penalties
advanced_team_id
```

### Skor semantigi

- `home_goals/away_goals`: 90 dakika veya uzatma oynandiysa 120 dakika field
  score.
- Shootout golleri bu kolonlara eklenmez.
- `decided_on_penalties=true`, field score sonucunu otomatik beraberlik yapmaz.
- `advanced_team_id` yalniz tie decider icin tie taraflarindan biri olmalidir.

### xG semantigi

- Iki xG birlikte bulunur veya ikisi de bostur.
- `xg_analysis_eligible=true` yalniz scope dogrulanmis iki tarafli xG'de olur.
- Shootout xG'ye eklenmez.
- Uzatma oynanan macta 90 dakika xG'si 120 dakika field score ile karistirilmaz;
  scope uyumsuzsa eligibility false olmalidir.

## 9. Dynamic State Output

`ratings_state.csv` kolonlari:

```text
season
team_id
team_name
ao_first_elo
power_elo
achievement_reserve
progression_bonus_ucl
progression_bonus_uel
progression_bonus_uecl
progression_bonus_total
ao_live_elo
last_event_utc
last_match_id
model_version
config_id
```

State satirinda `ao_live_elo`, bilesenlerden yeniden hesaplanabilir olmalidir.
Model ve config fingerprint uyusmazligi state devaminda reddedilir.

## 10. Match Update Audit

`match_updates.csv` her mac icin su gruplari saklar:

- kimlik, chronology, competition/stage/tie,
- qualification round key, stage K multiplier ve effective K,
- geriye uyumlu qualifier carry applied/adjustment audit'i; aktif continuous
  retention sozlesmesinde bu alanlar her zaman `false/0` olur,
- pre-match Power/Reserve/Progression/Live,
- effective rating farki, expected score ve H/D/A,
- field result ve GD parametre/multiplier,
- xG eligibility, fallback, signal ve adjustment,
- final Power Delta,
- progression recipient, requested/applied/cap bakiyesi,
- post-match tum rating bilesenleri,
- model version ve config ID.

Audit kolonlari, final ratingi yalniz toplu degerden degil bilesenlerden yeniden
uretmeyi mumkun kilmalidir.

Checkpoint metadata'si ayrica `qualification_participants` ve
`qualification_carry_applied` takim setlerini saklar. Ikinci set birincinin
alt kumesi olmalidir. Alan geriye uyumluluk icin korunur; aktif continuous
retention sozlesmesi MAIN gecisinde carry uygulamadigi icin ikinci set bostur.

## 11. Locked Prediction Ledger

Prospective degerlendirme icin prediction record su kosullari saglar:

- `generated_at_utc < kickoff_utc`,
- state chronology pointer'lari kayitli,
- sonuc/skor/xG/advanced team bulunmaz,
- pre-match component ratingleri ve 1X2 kayitlidir,
- schema/model/config fingerprint vardir,
- settlement ayni fixture ve state ile eslesir.

Ledger append-only olmalidir. Eski prediction sonuctan sonra overwrite edilmez.

### 11.0 Ledger uygulamasi

`src/ao_elo/prediction_ledger.py` bu sozlesmeyi hash zinciriyle uygular.
Dosya: `data/prediction_ledger/prediction_ledger_2026_27.jsonl`, satir basina bir
JSON kaydi:

```text
sequence        artan tam sayi
kind            PREDICTION | SETTLEMENT
recorded_at_utc kaydin yazildigi an
payload         servis edilen satirin tamami
previous_hash   bir onceki girisin entry_hash'i
entry_hash      sha256(canonical(sequence,kind,recorded_at,payload,previous_hash))
```

Yazma kurallari:

- `generated_at_utc < kickoff_utc` olmayan satir reddedilir.
- `recorded_at` kickoff'tan sonraysa reddedilir.
- Sonuc alanlari (`home_goals`, `outcome`, `xg_*` ...) pre-match kayitta bulunamaz.
- Bir mac kickoff'tan once **revize edilebilir**; eski surum silinmez,
  `ledger_revision` ile numaralanir ve son pre-kickoff surum kilitli tahmindir.
- Settlement ayri bir giris olarak eklenir; tahmin satirina dokunulmaz ve
  fikstur kimligi (kickoff, iki kulup) tahminle birebir eslesmelidir.
- Settlement yazildiktan sonra o mac icin yeni tahmin kabul edilmez.
- Toplu yazimda tek bir satir gecersizse hicbiri yazilmaz.

**Zincirin kanitladigi ve kanitlamadigi:** icerigi degistirmek girisin hash'ini,
girisin hash'ini duzeltmek de sonraki girisin `previous_hash`'ini bozar; ikisi de
`verify_ledger` tarafindan yakalanir. Fakat butun dosyayi yeniden yazabilen biri
tutarli bir zinciri uydurma zaman damgalariyla yeniden kurabilir -- bu durumda
zincir gecerli gorunur ama **head hash degisir**. Bu yuzden koruma ancak head
hash disarida sabitlendiginde tamamlanir. `ledger_anchor_2026_27.json` bu amacla
tutulur ve commit/push ile uzak deponun zaman damgasina baglanir.

### 11.1 Production Prediction Ensemble Input ve Logu

Aktif ensemble, base locked prediction'a ek olarak
`ml_features.FEATURE_SCHEMAS["STRUCTURAL_LOGISTIC"]` icindeki pre-match
feature'lari okur. Zorunlu kimlik/base alanlari:

```text
match_id
season
kickoff_utc
competition
round
stage
format_type
is_neutral
home_club_id
away_club_id
ao_home_probability
ao_draw_probability
ao_away_probability
```

#### `round` ve `round_sequence`: egitim vokabuleri baglayicidir

`round` ve `stage` kategorik, `round_sequence` ve `leg_number` sayisal
feature'lardir.  Kategorik encoder `handle_unknown="ignore"` ile kurulur: egitimde
gorulmemis bir deger **hata vermez**, one-hot'u sessizce sifirlanir.  Bu yuzden
servis edilen her `round` degeri egitim vokabulerinde bulunmak zorundadir.

UEFA'nin `GROUP` modlu lig asamasi egitim verisinde **`League Stage`** olarak
gecer (2024/25 ve 2025/26, 792 satir).  Saglayicinin `League phase` etiketi bu
ada cevrilir; `League Phase` uretmek modelin hic gormedigi bir kategori yaratir
ve o satirlar tur bilgisini tumden kaybeder.  `ROUND_NAME_MAP`'in urettigi her
ad icin bu kural testle pinlidir.

`round_sequence` sezon-ici bir siradir: o sezonun fiilen icerdigi turlar uzerinden
atanir, dolayisiyla oynanmakta olan bir sezon icin hesaplanamaz.  Servis edilen
fiksturlere sabit `-1.0` yazilirdi; bu deger hicbir egitim satirinda yoktur ve
model her tahminde uyduruldugu araligin disina cikardi.  Artik en son tamamlanmis
sezonun `(competition, round)` ordinalleri tasinir -- format degismedigi surece
dogru degerler bunlardir:

```text
League Stage    UCL 4.0    UEL 14.0    UECL 24.0
```

Eslenmemis bir tur artik `-1.0`'a dusmez, **hata verir**.  Bu iki kusur birlikte
servis edilen 1X2'yi `0.0019` kaydiriyordu; ikisi de sessizdi.

Feature store yalniz kickoff'tan once bilinen AO, format, Avrupa formu,
dinlenme ve mac yogunlugu alanlarini icerebilir. Hedef macin sonucu, skoru,
xG'si veya match-sonu ratingi girdi olamaz.

Production log semasi
`production_prediction.PRODUCTION_PREDICTION_LOG_COLUMNS` tarafindan
dondurulmustur. Her satir en az sunlari audit eder:

- AO, ham ML, Current ML, ham Poisson, AO Poisson ve final H/D/A olasiliklari,
- Domestic Poisson coverage ve component fallback,
- prediction status ile fallback reason,
- model/config/contract/manifest/ML/state SHA-256 kimlikleri,
- `rating_feedback_applied=false` invarianti.

Artifact manifesti, frozen ML modeli ve Domestic Poisson state'i:

```text
artifacts/production_prediction/manifest.json
artifacts/production_prediction/structural_logistic_v1.joblib
artifacts/production_prediction/domestic_poisson_state_2026_27.json
```

Artifact veya gerekli feature bozulursa servis sonucu tahmin etmez ya da sifir
doldurmaz; ayni satir icin final H/D/A'yi Current AO 1X2 olarak kaydeder.

2026/27 aktif checkpoint'i, UCL/UEL domestic coverage genisletmesinden gelen
79 uygun hedef kulubu ve sadece ilgili Avrupa kickoff'undan once tamamlanmis
yerel maclari icerir. União Torreense, iki sezon/40 mac esigini gecmedigi icin
Poisson `NONE/ONE` kurallarindan guvenli fallback alir.

Domestic engine checkpoint semasi `2.0`, `processed_event_ids` listesini ve
her lig icin `last_kickoff_utc` alanini saklar. Eski `1.0` checkpoint otomatik
migrate edilmez; kaynak sonuclardan yeniden uretilir. Event tekrar uygulama,
lig icinde geri tarih ve ayni lig/kickoff'u birden fazla batch'e bolme ret
sebebidir. Gecersiz batch state degismeden reddedilir.
Snapshot'in cutoff'u `<= generated_at_utc` ve `< fixture kickoff_utc` olmali;
aksi halde satir `FALLBACK_CURRENT_AO` olur. Bu kontrol sonucun gercekten ne
zaman edinildigini kanitlamaz; provider availability ve append-only ledger
ingestion katmaninin ayrica saglamasi gereken kosullardir.

### 11.2 Domestic kaynak sezonu ve fikstur uzlastirmasi

Secondary archive'de `provider_season`/`season` varsa kaynak sezonu kullanilir.
Yoksa takvim liglerinde kickoff'un UTC yili kullanilir; AO'nun Temmuz siniri
provider yilini degistirmez. Tarihsel `start_year/end_year` filtresi provider
sezonuna uygulanir. Kis liglerinde kaynak sezonu olmayan arsiv icin Temmuz
siniri tahmini kullanilir; acik kaynak sezonu gec oynanan maclarda onceliklidir.

Merge'den once lig/takim provider alias'lari kanoniklestirilir. Fikstur anahtari
`(sportsdb_league_id, normalized kickoff_utc, home_source_team_id, away_source_team_id)`
benzersizdir; farkli event ID veya AO sezonu ikinci mac yaratmaz. Merge ve replay
tekrarli fiksturu reddeder. Replay'de otomatik dedup yoktur.

Kaynak normalizasyonu ayni fikstur/ayni skoru tek gozleme indirirken her KEEP ve
REMOVE_OBSERVATION kararini `fixture_reconciliation_audit.csv`'ye yazar.
Celisen skorlar yalniz `data/domestic_fixture_reconciliations.json` icindeki
resmi kaynak karari ve birebir eslesen gozlemlerle uzlastirilir. Bilinmeyen
celiski butun build'i durdurur; sezon sessizce atlanmaz. Ham cache degismez.

Secondary coverage beklentisi en az iki kez gorulen moddur; hicbir sayim tekrar
etmiyorsa `ceil(median(season_counts))` kullanilir. Audit'teki
`format_expectation_method` bu ayrimi saklar. Her iki yontem de tahminidir;
resmi fikstur formati veya eksiksizlik kaniti degildir. Kabul esigi yine %95'tir.

#### Bilinen kapsam bosluklari

Kaynak sezonu duzeltildikten sonra iki lig-sezonu production girdisinin disinda
kalir. Bu karar `FROZEN_ACCEPTED_COVERAGE_GAPS` ile pinlidir; provider'in daha
sonra farkli bir tablo dondurmesi sezona sessizce production girisi vermez:

| Lig-sezon | Arsivdeki oynanmis mac | Beklenen (mod) | Coverage | Sonuc |
| --- | ---: | ---: | ---: | --- |
| GEO 2014 | 120 | 184 | 0.652 | REJECTED |
| LIT 2020 | 60 | 127 | 0.472 | REJECTED / FROZEN |

Ikisi de secondary arsivde tam olarak mevcuttur ve `is_played` degeri dogrudur;
kapi eksik veri gordugu icin degil, sezonun mac sayisi ligin tekrar eden format
sayisindan materyal olarak farkli oldugu icin reddeder. Bu iki sezon onceki
revision'da **yanlislikla** dahildi: bozuk provider-sezon turetmesi onlari gecerli
bir sezon kovasina karistiriyordu. Duzeltme etiketi dogrulttu, kapi da tasarlandigi
gibi calisti.

Yayimlanan `league_season_quality.csv` bu iki satiri
`REJECTED / UNAVAILABLE / FROZEN_ACCEPTED_COVERAGE_GAP` olarak gosterir. Kaynak
secimi audit'i diger provider'in kendi verdictini de korur. 2026-08-29
yenilemesinde primary LIT 2020 icin `60/60` kabul sonucu dondurdu; pin bu yeni
kaynak sonucunun daha once kabul edilmis coverage politikasini sessizce
degistirmesini engelledi. Secondary olcumu `60/127 = 0.472441` olarak audit'te
ayrica gorunur.

#### Kickoff timezone siniri

Normalize edilmis domestic CSV sinirinda `kickoff_utc` **acik offset tasimak
zorundadir**.  `merge_domestic_candidate` ve `select_causal_domestic_results`
artik `validators.require_utc_column`, causal cutoff ise
`require_utc_timestamp` kullanir; timezone-naive bir kolon reddedilir, farkli
offset'ler tek tek dogrulanip UTC'ye cevrilir.

Bu sinir onemlidir cunku `replay_domestic_poisson_state` naive timestamp'i zaten
dogru reddeder; merge onu once UTC'ye damgalayinca o guard etkisiz kaliyordu.
`select_causal_domestic_results` tarafinda ayni damgalama cutoff'u kaydirip
kickoff'tan sonra oynanmis bir sonucun tahmine girmesine yol acabilirdi.

Ham provider arsivinin kendi schema'sinda UTC olarak tanimlanmis
timezone-naive alani ayri bir adapter sozlesmesidir
(`normalize_secondary_fixtures`); o sinir bu kuralin disindadir.

#### Kapi kararinin kaynak-bagimliligi

Coverage kapisi bir eksiksizlik olcusu **degildir**: verdigi karar hangi
kaynagin sectirildigine baglidir.  Ayni denetimde ortaya cikan karsi ornek:

| Lig-sezon | Secondary | Primary | Production sonucu |
| --- | --- | --- | --- |
| GEO 2014 | `120/184 = 0.652` REJECTED | UNAVAILABLE | disarida |
| LIT 2020 | `60/127 = 0.472` REJECTED | `60/60 = 1.000` ACCEPTED | **frozen pin nedeniyle disarida** |
| GEO 2020 | `94/184 = 0.511` REJECTED | `92/72 = 1.278` ACCEPTED | **iceride** |

GEO 2020 secondary'nin kendi olcusune gore GEO 2014'ten **daha kotudur**
(`0.511 < 0.652`), ama production'a girer; cunku primary'nin kendi beklentisi
`72`, secondary'nin cikardigi lig formati ise `184`.  Iki sezonun dusmesinin
sebebi daha eksik olmalari degil, daha gevsek beklentiye sahip bir primary
fallback'lerinin olmamasidir.

Kaynak secimi artik kaybeden kaynagin kendi verdictini de yazar.  Her satirda
su bes alan bulunur (kabul edilen kaynak primary ise `secondary_*`, secondary
ise `primary_*` onekiyle):

```text
<other>_quality_status          ACCEPTED / REJECTED / ABSENT
<other>_quality_reason
<other>_schedule_matches
<other>_table_expected_matches
<other>_coverage_rate
```

Boylece LIT 2020 satiri frozen policy kararini tasirken
`secondary_quality_reason = SECONDARY_INFERRED_FORMAT_BELOW_95_PCT` ve
`secondary_coverage_rate = 0.472441` de gorunur; GEO 2020 satirinda ise
`table_expected_matches = 72` ile `secondary_table_expected_matches = 184`
yan yana durur.  Bu kolonlar 2026-08-29 expansion build'inde yayimlanan CSV'de
mevcuttur.

Bu tekil bir durum degildir: `league_season_quality.csv` icinde **26 ulke** ayni
lig icin birden fazla `table_expected_matches` degeri tasir.  Dolayisiyla
"coverage kapisindan gecti" ifadesi sezonun tam oldugunu gostermez.  Kapinin
kaynak-bagimsiz hale gelmesi, beklenen mac sayisinin resmi fikstur kaydindan
pinlenmesini gerektirir; bu yapilmamistir.

Mevcut kanitla "gercekten kisa sezon" ile "eksik arsiv" ayirt edilemez; ayrim
resmi lig fikstur kaydi gerektirir. Bu yuzden beklenen sayi elle pinlenmemis,
bosluk **bilinen ve kabul edilmis** olarak kayda gecirilmistir. Etki yalnizca
domestic Poisson gecmisidir: hicbir AO First parametresi, exposure cap'i,
katilim k'si veya served blend agirligi bu iki sezondan turetilmez. Iki sezon
2026/27'ye sirasiyla 12 ve 6 sezon uzaklikta oldugu icin `season_carry` ile
sonumlenmis halde tasinir.

Bu kabul bir kalite iddiasi degildir. Beklenen sayiyi resmi kaynaktan pinlemek
veya takim sayisindan turetmek dogru cozumdur; holdout kilidinden sonraki bakim
turuna birakilmistir. Kapi kaldirilmamali, esik dusurulmemelidir.

## 12. Veri Kalitesi Kontrol Listesi

Bir veri seti analize hazir sayilmadan once:

1. Anahtarlar benzersiz mi?
2. Tum match takimlari AO identity registry'de mi?
3. Exact UTC ve event order kararlı mi?
4. Ayni takim ayni anda iki macta gorunuyor mu?
5. Penalty ve field score ayrimi dogru mu?
6. Tie/stage/advanced metadata tutarli mi?
7. xG iki tarafli ve zaman kapsami uyumlu mu?
8. Gelecek sezon veya test sonucu feature history'ye sizmis mi?
9. Eksik deger fallback'i acikca audit ediliyor mu?
10. Ayni raw/cache ile cikti deterministik mi?
