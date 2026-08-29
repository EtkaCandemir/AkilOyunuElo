# 2026-08-29 Canli Veri Yenileme Raporu

## Sonuc

2026/27 domestic state, production prediction artifact zinciri, current-model
kanitlari, external benchmark, Avrupa on eleme replay'i ve gelecek fikstur
tahminleri yenilendi. Kullanici tarafindan verilen cutoff
`2026-08-29T00:00:00Z` olarak uygulandi. Model parametreleri degistirilmedi.
Production contract'larda yalniz artifact manifest SHA-256 alani guncellendi.

Yeni live checkpoint:

```text
62ffabe7bd78ed59844aa6e09c93e235c960de5d41453c4aaa51aa199c05ee8d
```

Manifest SHA-256:

```text
d577e9a883351a534e1be17a081f83f1248b65b2eb9791b994744832ea5bd021
```

Yenileme sonrasi tam test sonucu `1214 passed, 1 warning` olmustur. Git commit
veya push yapilmadi. Provider cache'i elle degistirilmedi.

## Cutoff ve servis edilecek fikstur listesi

Domestic adimlarindan sonra, backtest'lerden once asagidaki komut calistirildi:

```bash
python3 scripts/build_2026_27_preproduction_inputs.py --refresh
```

Ilk kosu provider verisini basariyla yeniledi fakat eski `342 completed / 86
upcoming` sabit sayi guard'inda durdu. Guard, pozitif satir sayisi, match-id
tekilligi ve completed/upcoming ayrikligi ile degistirildi; ayni cache uzerinden
offline tekrar kosusu `23/23` kalite kontrolunu gecti.

| Olcum | Sonuc |
| --- | ---: |
| Upcoming fikstur | 144 |
| Benzersiz match ID | 144 |
| Kanonik fikstur tekrari | 0 |
| Cutoff'ta veya once baslayan upcoming fikstur | 0 |
| En erken kickoff | `2026-09-08T16:45:00Z` |
| En gec kickoff | `2027-01-27T20:00:00Z` |
| Competition | UCL: 144 |
| Round | League Phase: 144 |
| Stage | LEAGUE: 144 |
| Knockout | False: 144 |

En erken kickoff cutoff'tan `10 gun 16 saat 45 dakika` sonradir; cutoff kapisi
gecmistir. Lig asamasi planlanmistir fakat henuz oynanmis lig-asamasi maci
yoktur. Son tamamlanmis Avrupa maci `2026-08-27T19:00:00Z` kickoff'lu qualifying
play-off macidir.

## Calistirilan zincir

Asagidaki uretim sirasi izlendi:

1. `build_domestic_league_dataset.py --refresh`
2. `build_ucl_uel_domestic_expansion.py --refresh`
3. `build_ucl_uel_live_domestic_state.py --refresh --cutoff-utc 2026-08-29T00:00:00Z`
4. `build_2026_27_preproduction_inputs.py --refresh`
5. `run_ml_1x2_backtest.py --blend-weight 0.9 --bootstrap-samples 4000`
6. `run_domestic_poisson_backtest.py --rebuild-domestic-surface --bootstrap-samples 4000`
7. `run_final_prediction_ensemble_backtest.py --prospective-poisson-weight 0.5 --bootstrap-samples 4000`
8. Tarihsel ve live production artifact build'leri, ayri temiz ve bos output
   root'larinda
9. Iki contract'taki artifact manifest hash'i, current evaluation/snapshot ve
   external benchmark
10. 2026/27 Q1/Q2/Q3/QPO replay'i, feature build'i ve production prediction

`--blend-weight 0.9`, `--prospective-poisson-weight 0.5` ve
`--rebuild-domestic-surface` atlanmadi. Gamma sensitivity kosulmadi; dolayisiyla
`--variance-penalty 0.5` gerektiren bir secim yuzeyi calistirilmadi.

## Veri hacmi: once ve sonra

| Veri/katman | Once | Sonra | Fark |
| --- | ---: | ---: | ---: |
| Base domestic mac | 45,419 | 46,822 | +1,403 |
| Base source takim | 508 | 531 | +23 |
| Base'e map edilen AO kulubu | 171 | 181 | +10 |
| Candidate domestic mac | 75,226 | 75,673 | +447 |
| Candidate source takim | 959 | 979 | +20 |
| Candidate'a map edilen AO kulubu | 120 | 122 | +2 |
| Live 2026/27 domestic sonuc | 1,101 | 1,303 | +202 |
| Live sonuc takimlari | 187 | 247 | +60 |
| Production identity map | 312 | 312 | 0 |
| Production eligible AO kulubu | 311 | 311 | 0 |
| Coverage gate ile dislanan | 1 | 1 | 0 |
| Checkpoint source matches | 76,327 | 76,976 | +649 |
| Tamamlanmis 2026/27 Avrupa maci | 342 | 428 | +86 |
| Upcoming 2026/27 Avrupa fiksturu | 86 | 144 | +58 |

Base domestic sete SCO'nun girmesi 1,403 satirlik artisi aciklar. Bu maclar
expansion tarafinda da ayni provider gozlemleri olarak mevcuttu; final candidate
sayisina iki kez eklenmedi.

Checkpoint'in `state_cutoff_utc` degeri son dahil edilen domestic sonucun kickoff
zamanidir ve `2026-08-28T19:30:00+00:00` olmustur. Bu alan kullanici cutoff'unun
kendisi degildir.

## Replay ve tahmin ciktisi

2026/27 replay'i artik tamamlanmis qualifying play-off turunu da kapsar:

| Replay olcumu | Once | Sonra |
| --- | ---: | ---: |
| Mac | 342 | 428 |
| Cutoff | Q3 sonrasi / QPO oncesi | QPO sonrasi / League Phase oncesi |
| Ortalama mutlak Elo degisimi | 14.708270 | 17.885805 |
| Medyan mutlak Elo degisimi | 10.322071 | 11.284267 |
| P95 mutlak Elo degisimi | 46.031908 | 57.484620 |
| Maksimum kazanc | 89.891140 | 88.472322 |
| Maksimum kayip | -52.766176 | -65.563535 |

Replay'de Q1/Q2/Q3/QPO mac sayilari `92/144/106/86`, efektif stage-K
carpanlari `0.20/0.275/0.35/0.425`tir. Maksimum mac-basi sifir-toplam hatasi
`1.1369e-13` olculdu.

Prediction feature bridge'i 6,768 tamamlanmis Avrupa macini kullandi ve 144
fiksturu 31 exact-UTC kickoff batch'inde uretti. Tahmin sonucu:

| Sayac | Deger |
| --- | ---: |
| ACTIVE_ENSEMBLE | 144 |
| Fallback | 0 |
| Domestic Poisson coverage BOTH | 144 |
| `rows_with_imputed_model_input` | 8 |

Sekiz imputed-input satiri fallback sayacina girmez; bunlar ACTIVE_ENSEMBLE
olarak servis edilir ve ayri izlenmelidir.

## Backtest metrikleri

Bu metrikler 2020/21-2025/26 nested/OOS development penceresine aittir; sabit
production ensemble'inin birebir prospective replay'i degildir.

| Katman | Olcum | Once | Sonra |
| --- | --- | ---: | ---: |
| Structural ML | Brier | 0.562750 | 0.563746 |
| Structural ML | Log-loss | 0.951215 | 0.952686 |
| Structural ML | Delta Brier vs AO | -0.003663 | -0.002666 |
| Structural ML | Delta log-loss vs AO | -0.005045 | -0.003573 |
| Domestic Poisson | Brier | 0.564201 | 0.564488 |
| Domestic Poisson | Log-loss | 0.953045 | 0.953443 |
| Domestic Poisson | Delta Brier vs AO | -0.002212 | -0.001925 |
| Domestic Poisson | Delta log-loss vs AO | -0.003214 | -0.002816 |
| Final nested ensemble | Brier | 0.562065 | 0.563050 |
| Final nested ensemble | Log-loss | 0.949965 | 0.951368 |
| Final nested ensemble | Accuracy | 0.561220 | 0.559582 |
| Final nested ensemble | Delta Brier vs AO | -0.004348 | -0.003362 |
| Final nested ensemble | Delta log-loss vs AO | -0.006294 | -0.004891 |

Kararlar degismedi: Structural ML `KEEP_SHADOW`, Domestic Poisson
`PROMOTE_CANDIDATE`, final tarihsel nested ensemble `KEEP_SHADOW`; production
prediction katmani manuel urun karariyla `PROMOTE_WITH_MONITORING` olarak
kaldi. Backtest Domestic Poisson yuzeyinin secimi `0.4 -> 0.3` degisti; bu
research secimidir. Servis edilen Poisson ic agirligi donmus `0.5`, `rho=0` ve
prospective final agirlik `0.5` olarak kalmistir.

External benchmark'ta 363 maclik AO rating-core kolu degismedi. Servis edilen
ensemble diagnostigi Brier `0.579166 -> 0.580864`, log-loss
`0.973577 -> 0.975477`, accuracy `0.553719 -> 0.556474` oldu.

## Karsilasilan veri/provider sorunlari ve cozumleri

1. **Base/expansion tekrarli SCO gozlemleri.** Base refresh SCO'yu kapsama
   aldiginda 1,403 ayni provider maci expansion'da da kaldi ve duplicate
   `match_id` guard'i build'i durdurdu. Iki taraftaki provider kimligi, kanonik
   fikstur, UTC kickoff ve skorlar birebir karsilastirildi. Yalniz tamamen ayni
   gozlemlerde base tercih edildi; kararlar
   `base_expansion_overlap_audit.csv`'ye yazildi. Herhangi bir fark halen hard
   error'dur.
2. **LIT 2020 provider sonucu degisti.** Primary kaynak bu kosuda `60/60`
   ACCEPTED dondurdu; onceki kabul edilmis coverage boslugunu sessizce tersine
   cevirecekti. `GEO 2014` ve `LIT 2020`
   `FROZEN_ACCEPTED_COVERAGE_GAPS` ile pinlendi. Final kalite etiketi
   `FROZEN_ACCEPTED_COVERAGE_GAP`; secondary LIT olcumu `60/127 = 0.472441`
   audit'te korunur.
3. **Live identity evreni genisledi.** Registry exact-name eslesmeleri coverage
   audit'inde hic degerlendirilmemis kulüpleri map ederek ara kosuda
   `324 mapped / 323 eligible` uretti. Yeni live eslesmeler mevcut production
   identity evreni ile acik UCL/UEL target evrenine sinirlandi.
4. **UKR provider isimleri kisaldi.** Provider `Dynamo Kyiv`, `Vorskla Poltava`
   ve `Zorya Luhansk` yerine `Dynamo Kiev`, `Vorskla`, `Zorya` dondurdu; ilk
   sinirli replay `309 mapped` oldu. Registry'de tekil oldugu dogrulanan uc
   provider alias'i pinlendi; final sayac `312/311/1` oldu.
5. **Preproduction sabit satir guard'i eskidi.** Refresh dogru bicimde 86 QPO
   sonucunu tamamlanmisa tasiyip 144 League Phase fiksturu getirdi; eski
   `342/86` guard'i bunu hata saydi. Sabit snapshot sayilari kaldirildi ve
   tekillik/ayriklik/pozitiflik guard'lari kondu.
6. **League Phase metadata'si knockout yaziliyordu.** UEFA `GROUP` modu daha
   once kosulsuz `is_knockout=True`, `stage=QUALIFYING` ve sentetik `tie_id`
   aliyordu. Round modu fail-closed ayrildi; `GROUP` artik `LEAGUE`, non-knockout,
   `tie_id=None`, `leg_number=0` uretir.

Live lig fetch asamasinda kurtarilmamis HTTP/parse hatasi olmadi; 34 lig cache'i
ve manifest'e girdi. Uc mevcut skor uzlastirma karari degistirilmedi ve yeni bir
manuel skor karari eklenmedi.

## Artifact ve hash zinciri

| Artifact | SHA-256 |
| --- | --- |
| Historical domestic state 2025/26 | `92b74f2bdf3826c7e1a7732aa11f33eb509bf401a0931d8acf8ae5d73408f120` |
| Live domestic state 2026/27 | `62ffabe7bd78ed59844aa6e09c93e235c960de5d41453c4aaa51aa199c05ee8d` |
| Structural Logistic joblib | `846f48a80fbd9c237aa4667ad773f41ba970c71251fda3aad357ce41a8fcb09a` |
| Production manifest | `d577e9a883351a534e1be17a081f83f1248b65b2eb9791b994744832ea5bd021` |

Iki contract da ayni manifest hash'ini tasir. Manifest'in ML ve state hash'leri,
state'in historical/live/bridge/coverage input hash'leri dosyalardan bagimsiz
hesaplanarak dogrulandi. Strict production runtime ayni manifest hash'iyle
yuklendi.

Current-model raporundaki contract `prediction_layer_evidence` sayilari
aktivasyon zamaninda donmus kanittir; bu yenilemenin backtest metrikleriyle
degistirilmedi. Rapor metni bu iki kanit sinifini birbirine kaynak gostermeyecek
sekilde duzeltildi.

## Degismeyen parametre ve AO cekirdegi

| Donmus deger | Dogrulanan |
| --- | ---: |
| `max_european_exposure` | 0.65 |
| `unknown_league_finish_score` | 0.15 |
| European participation k | 0.20 |
| `cup_contribution_weight` | 0.12903225806451613 |
| Efektif servis AO/ML/Poisson | 0.30 / 0.45 / 0.25 |
| Poisson ic agirlik / rho | 0.5 / 0.0 |
| ML blend | 0.9 |

AO First/rating core yeniden hesaplamasinda asagidaki dosyalar onceki snapshot
ile byte-identical kaldi:

| Dosya | SHA-256 |
| --- | --- |
| `model_summary.csv` | `2b01071a16d72edaa95eed895f629482975d61b7c3b943a8bf5168d47a35ba78` |
| `fold_summary.csv` | `ff2711a98d0e0893b6e411caa84f009b6d3fdc9feb20c22021fc5ba551ec5742` |
| `competition_summary.csv` | `d7497ca8ce2215b54615c8a31783aa45bd88cbbee8c6eee79837d7d31701dce6` |

## Zorunlu dogrulamalar

- Baslangic testi: `1214 passed, 1 warning`.
- Final testi: `1214 passed, 1 warning`.
- Bagimsiz `merge_domestic_candidate + replay_domestic_poisson_state` replay'i:
  yayimlanan `engine_state` ile JSON payload birebir esit.
- Kanonik fikstur tekrari: `0`.
- `processed_event_ids`: `76,976`, benzersiz ve `source_matches=76,976` ile esit.
- `production_eligible_ao_clubs=311`, `excluded_by_coverage_gate=1`.
- `_secondary_cache/fixtures.parquet` icindeki `is_played` dtype: `bool`.
- Uc `domestic_team_bridge*.csv` icin ambiguous ve map edilmis satir: `0`.
- Contract/manifest/artifact/input hash zinciri: PASS.
- Strict runtime load: PASS.
- Production prediction state'i rating state'ine geri beslenmedi.

Tek test warning'i joblib'in fiziksel CPU cekirdek sayisini okuyamamasindan
kaynaklanir; test sonucu veya artifact hesaplamasini degistirmedi.

Tekrar uretilebilir bagimsiz kontrol:

```bash
python3 output/live_refresh_2026_08_29/validate_refresh.py
```

Sonuc dosyasi:
`output/live_refresh_2026_08_29/refresh_validation_results.json`.

## Olcum sinirlari

- 144 tahmin icin mac sonucu henuz yoktur; prospective Brier/log-loss veya
  kalibrasyon olculmedi.
- Bu prediction CSV'si tekrar uretilebilir bir release ciktisidir; append-only,
  degistirilemez prospective ledger ve harici zaman damgasi kaniti degildir.
- Sekiz imputed-input satirinin sonuc etkisi ayri bir counter olarak raporlandi;
  outcome olmadigi icin performans etkisi olculmedi.
- Coverage gate provider kaynaklarinin eksiksizligini kanitlamaz; format
  beklentisi mod/median tahminidir. GEO 2014 ve LIT 2020 kabul edilmis frozen
  kapsam bosluklari olarak disarida kalir.
- External benchmark 363 eslesmis macla sinirlidir ve sabit production
  50/50 ensemble'inin 2026/27 prospective kaniti degildir.
- Bu repo production ingestion servisi degildir; fetch worker, kalici DB,
  identity servisi, prediction lock queue, retry/rate-limit/alarmlar ve read API
  bu yenilemeyle kurulmadı veya test edilmedi.
- Git temizligi, deploy, commit ve push olculmedi/yapilmadi. Kullaniciya ait
  mevcut degisiklikler geri alinmadi.
