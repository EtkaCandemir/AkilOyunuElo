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

## 11. Locked Prediction Ledger

Prospective degerlendirme icin prediction record su kosullari saglar:

- `generated_at_utc < kickoff_utc`,
- state chronology pointer'lari kayitli,
- sonuc/skor/xG/advanced team bulunmaz,
- pre-match component ratingleri ve 1X2 kayitlidir,
- schema/model/config fingerprint vardir,
- settlement ayni fixture ve state ile eslesir.

Ledger append-only olmalidir. Eski prediction sonuctan sonra overwrite edilmez.

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
artifacts/production_prediction/domestic_poisson_state_2025_26.json
```

Artifact veya gerekli feature bozulursa servis sonucu tahmin etmez ya da sifir
doldurmaz; ayni satir icin final H/D/A'yi Current AO 1X2 olarak kaydeder.

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
