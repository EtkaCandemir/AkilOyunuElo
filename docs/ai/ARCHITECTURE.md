# Sistem Mimarisi

Bu belge AO European Elo'nun calisma zamanindaki katmanlarini, state
sahipligini ve moduller arasi sinirlari tanimlar. Aktif parametre otoritesi
`contracts/ao_european_elo_v2_production.json` dosyasidir.

## 1. Mimari Ilkeler

1. **Static ve dynamic ayridir.** AO First Elo sezon basinda uretilir; mac
   motoru bunu seed olarak okur.
2. **Tahmin ve settlement ayridir.** Sonuc gorulmeden once olasilik kilitlenir,
   sonra ayni state uzerinde sonuc islenir.
3. **Power ve achievement muhasebesi ayridir.** Power guncellemesi sifir
   toplamli, progression winner-only'dir.
4. **Prediction ve rating state ayridir.** Aktif ML + Domestic Poisson katmani
   olasiligi iyilestirir ama AO Live Elo'ya geri beslenmez. Diger research
   modelleri production contract degismeden aktif sayilmaz.
5. **Veri eksikligi sessizce sifir yapilmaz.** Avrupa gecmisi olmayan takim
   icin bile acik sifir history satiri zorunludur.

## 2. Katmanlar

### Katman A: Veri ve Kimlik

Girdiler `team_id`, `season` ve `country_code` uzerinden birlestirilir.
`validators.py` duplicate, eksik anahtar, gecersiz boolean, negatif/sonsuz
deger, tutarsiz lig sirasi ve eksik kulup-history satirlarini reddeder.

Kulup kimligi sezonlar arasinda kalici olmalidir. UEFA veya veri saglayici
kimlikleri audit alani olabilir; AO `team_id` state anahtaridir.

### Katman B: AO First Elo

`pipeline.compute_ao_first_elo()` dort tabloyu birlestirir ve su alt
bilesenleri hesaplar:

- bes sezonluk ulke puani ve league strength,
- yerel lig/kupa basarisi,
- Domestic Prior,
- kulubun bes sezonluk Avrupa puani ve European Prior,
- sezon ve mac kanitindan European Exposure,
- exposure ile prior blend,
- bes sezon varyans kontrollu Domestic Surprise,
- AO First Elo ve baslangic sirasi.

Bu katman yan etkisiz DataFrame donusumudur. Dynamic state yaratmaz.

### Katman C: Season State

`dynamic.initialize_season()` her takim icin `TeamRating` olusturur:

```text
ao_first_elo
power_elo
achievement_reserve
progression_bonus_ucl
progression_bonus_uel
progression_bonus_uecl
last_event_utc
last_match_id
```

Aktif `power_carry=0` oldugu icin sezon basi `power_elo=ao_first_elo` olur.
Progression bakiyeleri sifirdan baslar. Achievement Reserve aktif degildir.

### Katman D: Pre-Match Prediction

`lock_prediction()` mevcut `AO Live Elo` degerlerini okur, ev sahibi avantajini
yalniz normal sahada ekler ve `E_home` hesaplar. Beraberlik modeli `E_home`
degerini H/D/A olasiliklarina ayirir.

Tek maclik eleme formati fixture metadata'sindan bilinmelidir. Bu formatta
`draw_at_even=0.12`, diger maclarda `0.24` kullanilir. Bu secim rating farkini
ve beklenen puani degistirmez.

### Katman D2: Production Prediction Ensemble

`production_prediction.py`, base AO 1X2'yi iki prediction-only bilesenle
isler:

1. Structural Logistic olasiligi `%90`, AO olasiligi `%10` ile Current ML.
2. `rho=0` Domestic Poisson olasiligi `%50`, AO olasiligi `%50` ile AO Poisson.
3. Bu iki bilesen log-probability uzayinda `%50/%50` birlestirilir.

Artifact ve state SHA-256 degerleri yukleme sirasinda dogrulanir. Hata veya
eksik feature satirinda cikti birebir base AO 1X2'ye doner. Ensemble ciktisi
settlement veya Power Delta hesabina girmez.

### Katman E: Match Settlement

`update_match()` sonucu isler:

1. 90/120 field score'dan `S_home` belirlenir.
2. Kontrollu gol farki carpani hesaplanir.
3. Uygun iki tarafli xG varsa bounded xG performans duzeltmesi eklenir.
4. Home Power `+Delta`, Away Power `-Delta` degisir.
5. Match/tie audit state'i guncellenir.
6. Tie bittiyse ve stage uygunsa kazanana progression bonusu eklenir.

Power update sirasinda progression bakiyesi kullanilan pre-match Live ratingin
bir parcasidir; yeni progression bonusu ancak tie bittikten sonraki maclarda
tahmini etkiler.

### Katman F: Raporlama ve Evaluation

Evaluation kodu production motorunu yeniden oynatir. Ana model ve ablation
kollari ayni fixtures, chronology ve baslangic evrenini kullanir. `output/` ve
`reports/` altindaki artefact'lar calisan kodun ciktilaridir; kaynak otoritesi
degildir.

## 3. State Formulu

```text
Progression Total = bonus_ucl + bonus_uel + bonus_uecl
AO Live Elo = Power Elo + Achievement Reserve + Progression Total
```

Aktif durumda:

```text
Achievement Reserve = 0
AO Live Elo = Power Elo + Progression Total
```

Bu nedenle iki conservation kavramini ayirmak gerekir:

- `Power Elo`: her macta ve sezon icinde korunur.
- `AO Live Elo`: winner-only progression nedeniyle toplamda artabilir.

## 4. Modul Sahipligi

| Modul | Sahip oldugu davranis | Sahip olmadigi davranis |
| --- | --- | --- |
| `config.py` | Static parametreler | Runtime chronology |
| `validators.py` | Input schema ve semantik validation | Rating formulu |
| `features.py` | Ulke, domestic, history, exposure feature'lari | Final blend |
| `scoring.py` | Norm, prior ve exposure blend | CSV birlestirme |
| `pipeline.py` | AO First orchestration ve output invariantlari | Match updates |
| `dynamic.py` | State, prediction lock, chronology, settlement | Static feature toplama |
| `draw_probability.py` | Score-preserving 1X2 | Power update |
| `controlled_live.py` | Goal-margin multiplier | xG eligibility |
| `xg_live.py` | xG ve birlesik Power Delta kernel'i | Progression |
| `tournament_bonus.py` | Stage uygunlugu, oran ve cap | Match sonucu Delta |
| `production_prediction.py` | Aktif ML+Poisson tahmini, artifact dogrulama, AO fallback | Rating update |
| `domestic_poisson.py` | Causal yerel attack/defence state ve snapshot | AO Live mutation |

## 5. Public Arayuzler

Static:

```python
compute_ao_first_elo(...)
compute_ao_first_elo_from_csv(...)
```

Dynamic:

```python
expected_score(...)
expected_1x2_probabilities(...)
initialize_season(...)
lock_prediction(...)
settle_locked_match(...)
update_match(...)
run_season(...)
ProductionPredictionService.from_contract(...)
ProductionPredictionService.predict(...)
```

CLI/batch katmani `dynamic_csv.py` bu cekirdegi kullanmalidir. Ayrica formul
yeniden yazmak drift riski yaratir.

## 6. Production ve Research Siniri

Asagidaki modullerin repository'de bulunmasi production aktivasyonu anlamina
gelmez:

- `prediction_ensemble.py` ve backtest orchestration kodu
- `model_based_partitioning.py`
- `opponent_quintile_context.py`, `relative_opponent_profile.py`
- `team_venue_context.py`, `match_context.py`, `dynamic_k.py`
- `scoreline.py`, `scoreline_calibration.py`

Bu moduller challenger, diagnostic veya shadow calismalarina aittir.
`domestic_poisson.py` ve `ml_prediction.py` yalniz
`production_prediction.py` icindeki checksum'lu artifact yolu uzerinden aktif
prediction'a katilir. Production davranisi contract + runtime loader +
production testleriyle belirlenir.

## 7. Degisiklik Etki Matrisi

| Degisiklik | Guncellenmesi gerekenler |
| --- | --- |
| Static formul/parametre | config, pipeline testleri, contract, snapshot, AO First regression |
| Dynamic Scale/H/K | dynamic config, replay, calibration, contract, zero-sum testleri |
| 1X2 modeli | draw module, locked prediction schema, Brier/log-loss evaluation |
| GD/xG | update kernel, match audit kolonlari, replay ve conservation testleri |
| Progression | stage normalizasyonu, tie audit, cap/reset testleri, contract |
| Veri semasi | validator, README/data contract, fixtures ve migration |
| Research terfisi | incremental ablation, prospective risk, kullanici onayi, production contract |

Canli veri toplama, prediction lock ve settlement servislerinin operasyonel
sozlesmesi [LIVE_DATA_INGESTION.md](LIVE_DATA_INGESTION.md) icindedir.
