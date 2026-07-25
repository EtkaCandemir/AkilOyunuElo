# AO European Elo

AO European Elo, UEFA kulüp turnuvaları için sezon öncesi başlangıç rating'i,
exact-UTC maç olaylarıyla güncellenen canlı Power Elo ve standart H/D/A maç
olasılıkları üreten açıklanabilir bir Python/Pandas motorudur.

Aktif geliştirme sözleşmesi `ao-european-elo-v2.0-dev-freeze` sürümüdür.
Parametreler 2018/19-2025/26 geliştirme verisinde nested walk-forward ile
seçilmiştir. 2026/27 elemeleri freeze tarihinden önce başladığı için sezonun
tamamı untouched sayılamaz. Prospective holdout, yalnızca sonuçsuz fixture ile
maçtan önce kilitlenen 2026/27 lig aşaması ve sonrası kayıtlarından oluşacaktır.
Bu nedenle model henüz tamamen kanıtlanmış production modeli olarak tanımlanmaz.

## Model Özeti

Model üç değeri ayrı tutar:

```text
AO First Elo            sezon öncesi statik başlangıç gücü
Power Elo               maçlarla sıfır toplamlı değişen güç
Achievement Reserve     tur/kupa başarısı için ayrı rezerv

AO Live Elo = Power Elo + Achievement Reserve
```

Aktif sürümde Achievement Reserve backtest kapısını geçmediği için sıfırdır;
AO Live Elo ile Power Elo aynıdır.

### AO First Elo v2

```text
M = 1500 / (903.92 - 500) = 3.713606654783126
AO Elo v2 = 500 + M x (AO Elo v1.1 - 500)
```

`500-2000` bir referans bandıdır ve hesap sonunda clipping uygulanmaz. Bununla
birlikte aktif beta değerleri `0/0/0` iken bounded statik bileşenlerin ulaşılabilir
teorik üst sınırı `2000`dir. Yani aktif AO First Elo için 2000 aynı zamanda
yapısal bir saturation sınırıdır; Dynamic Power/Live Elo ise kırpılmaz ve bu
değeri aşabilir. Affine dönüşüm v1.1 takım sırasını korur. Dynamic Scale, saha
avantajı ve K aynı çarpanla ölçeklendiği için görünür puan farkının büyümesi tek
başına tahmin gücünü yapay biçimde değiştirmez.

```text
Domestic Prior =
    500
    + 519.9049316696 x League Strength
    + 594.1770647653 x Domestic Achievement x Achievement Scale

European Prior =
    500 + 1559.7147950089 x European History Norm

AO First Elo =
    Domestic Prior
    + Effective European Exposure x (European Prior - Domestic Prior)
```

Ham exposure oynanan sezon ve maç miktarından hesaplanır. Aktif effective
exposure tavanı `0.85`tir; tam Avrupa kanıtında Domestic Prior'ın yüzde 15'i
korunur. Avrupa geçmişi olmayan açık sıfır satırında final rating doğrudan
Domestic Prior'dır.

### Dynamic Power Elo

```text
E_home = 1 / (1 + 10 ^ -((Home Live - Away Live + H) / Scale))

Scale = 835.5614973262
H     = 148.5442661913      nötr sahada 0
K     = 103.9809863339

Delta = K x (S_home - E_home)
Home Power New = Home Power Old + Delta
Away Power New = Away Power Old - Delta
```

Season carry, standart 1X2 nested walk-forward testinde yalnızca `5/6` unseen
fold kazandığı için kapalıdır:

```text
Power Start New = Current AO First Elo
Power Carry = 0
```

Skor 90 dakika veya uzatma oynandıysa 120 dakika sonundaki saha skorudur.
Penaltı atışı golleri skora eklenmez; penaltıyla sonuçlanan saha beraberliği
normal Power Elo'da `S=0.5` kalır.

### Standart 1X2 Olasılıkları

`E_home` bir galibiyet olasılığı değil, normalize beklenen maç puanıdır. Aktif
olasılık katmanı bu değeri gerçek üç sınıflı çıktıya çevirir:

```text
P_draw = 0.24 x (4 x E_home x (1-E_home)) ^ 1.00
P_home = E_home - 0.5 x P_draw
P_away = 1 - E_home - 0.5 x P_draw
```

Bu dönüşümde olasılıklar toplamı `1` ve
`P_home + 0.5 x P_draw = E_home` eşitliği zorunludur. Standart Brier, H/D/A
sınıflarındaki üç kare hatanın toplamıdır; standart log-loss gerçekleşen sınıfa
atanan olasılığın negatif logaritmasıdır.

## Aktif ve Kapalı Katmanlar

| Katman | Karar | Aktif değer |
| --- | --- | ---: |
| Country/European/Exposure tail | `NO_PROMOTION` | beta `0/0/0` |
| Dynamic Scale/H/K | `PROMOTE` | `835.561/148.544/103.981` |
| Season carry | `DISABLE` | `0.00` |
| Standart 1X2 çıktı | `PROMOTE` | draw `0.24`, shape `1.00` |
| Kontrollü gol farkı | `PROMOTE` | `alpha=0.10`, `tau=300`, GD cap `4` |
| Sıfır-toplamlı tur bonusu | `REJECT` | aktif base bonus `0` |
| Achievement Reserve | `DISABLE` | base `0` |
| Normal maç turnuva K çarpanı | `DISABLE` | uygulanmaz |

Final robustness'taki eski LOG/SQRT gol farkı ailesi `3/6` Brier ve `2/5`
forward-ranking sonucu nedeniyle kapalı kalmıştır. Onun yerine 23 Temmuz
kontrollü deneyinde `1 + alpha x ln(min(GD,4)) x exp(-abs(D)/tau)` formülü
test edilmiştir. Önceden belirlenen `0.10/300` adayı Brier'ı `6/6` unseen
foldda iyileştirmiş, dependency zarfları tamamen negatif ve pooled sıralama
farkları pozitif gelmiştir. Manuel production kararıyla bu kontrollü katman
aktif edilmiştir. Tur bonusu, turnuva K ve Achievement Reserve kapalıdır.

Aktif runtime manifesti:

```text
contracts/ao_european_elo_v2_production.json
```

## Proje Yapısı

```text
src/ao_elo/
  config.py           statik v1.1/v2 parametreleri
  features.py         statik feature hesapları
  scoring.py          prior, exposure ve final rating formülleri
  validators.py       statik CSV validation
  pipeline.py         AO First Elo dataframe/CSV motoru
  dynamic.py          reusable canlı Elo Python API'si
  dynamic_csv.py      CSV sözleşmeleri ve batch çekirdeği
  evaluation.py       adjusted ranking, 1X2 ve belirsizlik metrikleri
  robustness.py       azalan getirili gol ve joint competition-K adayları
  controlled_live.py  kontrollü GD ve sıfır-toplamlı progression çekirdeği
  goal_shadow.py      bağımsız çok-kollu prospective shadow state'i

contracts/
  ao_european_elo_v2.json

data/
  pilot_10_teams/                 sentetik statik pilot
  pilot_20_teams/                 UCL/UEL/UECL kontrollü canlı Elo pilotu
  real_pilot_10_teams/            kaynaklı gerçek 10 takım pilotu
  dynamic_pilot/                  sentetik exact-UTC canlı maç akışı
  backtest_stage_b_2018_2026/     statik geliştirme verisi
  external_elo_benchmark_2018_2026/

scripts/
  run_v2_ranking_calibration.py
  run_v2_dynamic_calibration.py
  run_v2_goal_margin_calibration.py
  run_v2_achievement_reserve_calibration.py
  run_v2_evaluation_upgrade.py
  run_final_robustness.py
  run_controlled_goal_progression_backtest.py
  run_goal_shadow_parameter_search.py
  run_goal_difference_shadow.py
  run_v2_pilots.py
  run_pilot_20_teams.py
  build_pilot_20_teams_pdf.py
  run_dynamic_elo.py
  run_dynamic_live.py

tests/
output/
docs/
```

## Kurulum ve Test

```bash
python3 -m pip install -r requirements.txt
pytest -q
```

V1.1 sayısal pilotları ve API'si regresyon modeli olarak korunur. Yeni testler
v2 affine dönüşümünü, tail sürekliliğini, adjusted ranking guardrail'lerini,
exact-date dinamik çekirdeği, carry, standart 1X2, bağımlılık bootstrap'ı, gol
farkı, reserve, CSV/API parity ve frozen contract uyumunu kapsar.

## Statik Veri Sözleşmesi

AO First Elo dört CSV bekler:

1. `teams.csv`
2. `country_coefficients.csv`
3. `domestic_context.csv`
4. `club_european_points.csv`

Anahtarlar:

```text
teams.csv                    team_id
country_coefficients.csv     season + country_code
domestic_context.csv         season + team_id
club_european_points.csv     season + team_id + country_code
```

Her hedef takım için kulüp geçmişi satırı zorunludur. Avrupa geçmişi yoksa beş
sezon puan, played ve matches açıkça `0` yazılır. Duplicate anahtar, eksik veya
negatif puan, NaN/sonsuz değer, geçersiz boolean, lig pozisyonu çelişkisi ve
pozitif olmayan match cap açıklayıcı hatayla reddedilir.

Statik output'ta `ao_first_elo`, deterministik `ao_first_elo_rank`, uncapped ve
tail-adjusted normlar, saturation/tail bayrakları, Domestic/European prior ve
iki exposure alanı bulunur.

## Pilotları Çalıştırma

V1.1 regresyon pilotları:

```bash
python3 scripts/run_pilot_10_teams.py
python3 scripts/run_real_pilot_10_teams.py
```

V2 statik ve canlı uçtan uca pilot:

```bash
python3 scripts/run_v2_pilots.py
```

20 takımlı kontrollü gol farkı production pilotu:

```bash
python3 scripts/run_pilot_20_teams.py
python3 scripts/build_pilot_20_teams_pdf.py
```

Bu sentetik paket `8 UCL / 6 UEL / 6 UECL` takım ve 33 maç içerir. Başlangıç
rating'leri `940-1940` aralığında ayrı CSV'dedir. Production replay yanında gol
farkı kapalı karşı-olgusal replay de üretilir; böylece final Elo üzerindeki net
gol farkı etkisi takım bazında ölçülür.

Ana çıktılar:

```text
output/pilot_20_teams/team_start_end_summary.csv
output/pilot_20_teams/match_updates_detailed.csv
output/pilot_20_teams/scenario_summary.csv
output/pilot_20_teams/competition_summary.csv
output/pilot_20_teams/AO_European_Elo_20_Takim_Pilot_Raporu.pdf
```

Üretilen ana dosyalar:

```text
output/v2_pilots/synthetic_ao_first_elo_v2.csv
output/v2_pilots/real_ao_first_elo_v2.csv
output/v2_pilots/dynamic_replay/ratings_state.csv
output/v2_pilots/dynamic_replay/state_checkpoint.json
output/v2_pilots/dynamic_replay/match_updates.csv
output/v2_pilots/dynamic_replay/replay_predictions.csv
output/v2_pilots/pilot_report.md
```

Gerçek 10 takım pilotunda sıra:

```text
1 Arsenal             1992.870
2 Sporting CP         1926.711
3 Benfica             1881.335
4 Shakhtar Donetsk    1764.684
5 Galatasaray         1756.282
6 AZ Alkmaar          1741.804
7 Slavia Praha        1621.856
8 Pafos               1444.036
9 Como                1421.436
10 Omonia Nicosia     1398.415
```

Bu sıra v1.1 pilotuyla birebir aynıdır. Como sıfır exposure kontrolü olarak
v2 Domestic Prior'ını korur ve yapay biçimde zirveye çıkmaz.

## Dinamik CSV Motoru

### Retrospective Replay

`matches.csv` çekirdek alanları:

```text
match_id, season, kickoff_utc, competition, round,
home_team_id, away_team_id, home_goals, away_goals,
is_neutral, decided_on_penalties
```

Eleme turları için `tie_id`, `is_knockout`, `is_tie_decider`, `stage` ve
`advanced_team_id` metadata'sı kullanılır.

```bash
python3 scripts/run_dynamic_elo.py \
  --initial-ratings output/v2_pilots/dynamic_initial_ratings.csv \
  --matches data/dynamic_pilot/matches.csv \
  --output-dir output/replay_run
```

Bu komut tamamlanmış sonuçları geriye dönük oynatır. Ürettiği tahmin satırları
maç öncesi değerleri gösterse de sonuç dosyası aynı çalıştırmada mevcut olduğu
için prospective holdout kanıtı değildir.

- `ratings_state.csv`: okunabilir takım snapshot'ı.
- `state_checkpoint.json`: processed maçlar, açık eşleşmeler, global kronoloji
  ve ratings checksum'u ile eksiksiz resumable state.
- `match_updates.csv`: pre/post rating, normalize beklenen puan, H/D/A
  olasılıkları, skor, delta ve reserve audit.
- `replay_predictions.csv`: açıkça `RETROSPECTIVE_REPLAY` etiketli tahmin audit'i.
- `batch_manifest.json`: `prospective_holdout_evidence=false` bildirimi.

### Prospective Canlı Akış

Gerçek maç öncesi kayıt iki aşamalıdır. `fixtures.csv` skor içermez; önce tahmin
kilitlenir, maç bittikten sonra ayrı `matches.csv` satırıyla settle edilir:

```bash
python3 scripts/run_dynamic_live.py initialize \
  --initial-ratings initial_ratings.csv \
  --state-dir output/live_state

python3 scripts/run_dynamic_live.py lock \
  --state-dir output/live_state \
  --fixture next_fixture.csv

python3 scripts/run_dynamic_live.py settle \
  --state-dir output/live_state \
  --result completed_match.csv
```

`pre_match_log.csv` append-only SHA-256 hash zinciri taşır, `generated_at_utc`
kickoff'tan önce olmak zorundadır ve eski satırlar yeniden yazılmaz. Hash zinciri
değişikliği görünür kılar; tek başına harici güvenilir zaman damgası değildir.
Eşzamanlı maçların tamamı kickoff öncesi aynı state'ten kilitlenebilir; sonuçlar
`kickoff_utc`, ardından `match_id` artan sırasıyla settle edilir. İlgili takımın
rating'i kilitten sonra değişmişse settlement reddedilir.

`expected_home_score`, ev sahibi galibiyet olasılığı değildir. Anlamı normalize
maç puanıdır: galibiyet `1`, beraberlik `0.5`, mağlubiyet `0`. Canlı ledger ve
replay çıktıları ayrıca `home_win_probability`, `draw_probability` ve
`away_win_probability` kolonlarını verir. Eski kalibrasyon raporlarındaki
`Brier`/`log-loss` adları legacy expected-score tanımlarını ifade eder; yeni
model seçiminde standart üç sınıflı `brier_1x2` ve `log_loss_1x2` kullanılır.

Duplicate maç, kronoloji gerilemesi, eksik takım, geçersiz skor, tie kimliği
çelişkisi ve config/state uyuşmazlığı reddedilir. Aynı state, config ve input
deterministik aynı sonucu üretir.

## V2 Kalibrasyonlarını Yeniden Çalıştırma

Sıra önemlidir; her optional katman seçilmiş bir önceki model üzerinde
incremental test edilir:

```bash
python3 scripts/run_v2_ranking_calibration.py
python3 scripts/run_v2_dynamic_calibration.py
python3 scripts/run_v2_goal_margin_calibration.py
python3 scripts/run_v2_achievement_reserve_calibration.py
python3 scripts/run_v2_evaluation_upgrade.py
python3 scripts/run_final_robustness.py
python3 scripts/run_controlled_goal_progression_backtest.py
python3 scripts/run_goal_shadow_parameter_search.py
```

Ana sonuçlar:

```text
Tail grid                 100 aday, 6 fold, NO_PROMOTION
Dynamic core              6/6 fold, expected-score MSE delta -0.003955
Legacy carry 0.85         expected-score MSE ile 6/6
1X2 carry seçimi          5/6 fold, DISABLE, aktif carry 0
Standart 1X2 çıktı        PROMOTE, draw 0.24 / shape 1.00
Goal margin final         3/6 Brier, forward ranking 2/5, DISABLE
Competition K final       1/6 Brier, forward ranking 1/5, DISABLE
Achievement Reserve final 2/6 Brier, forward ranking 1/5, DISABLE
Controlled GD nested       ΔBrier -0.000121, ranking 3/5, SHADOW
Progression-only nested    ΔBrier -0.000014, ranking 3/5, REJECT
Controlled full nested     ΔBrier -0.000260, ranking 1/5, SHADOW
Extended GD nested         ΔBrier -0.000309, SHADOW_EVIDENCE_COLLECTION
```

Yeni sıralama hedefi aday rating'lerinden bağımsızdır: saha etkisi takım dışı
maçlardan, rakip gücü doğrudan eşleşmeler dışarıda bırakılarak hesaplanır. Üç
belirsizlik görünümü tie/match, team-season ve calendar-month bootstrap'tır;
terfi kararı en geniş güven aralığı zarfını kullanır. Dinamik çekirdek standart
1X2 ile `6/6` doğrulanmıştır. Carry toplamda yararlı görünse de katı `6/6`
sözleşmesini geçmediği için üretimde kapalıdır.

Final robustness seçiminde dinamik sezon sonu rating'i aynı sezon sonuçlarına
karşı ölçülmez. Rating yalnız takip eden sezonun rakip ve saha etkisinden
arındırılmış performansına karşı değerlendirilir. Son 2025/26 rating'i için
2026/27 kullanılmadığından beş forward ranking fold'u vardır.

## Gol Farkı Prospective Shadow Akışı

Kontrollü `0.10/300` gol farkı production'da aktiftir. Dört ön-kayıtlı kol,
aktif production adayını gol farksız taban ve daha güçlü alternatiflerle
karşılaştırmak için aynı fixture ve sonuçları bağımsız state'lerde işler:

```text
BASE              0.000 / 300
PRE_SPECIFIED     0.100 / 300
PRIOR_GRID_BEST   0.200 / 400
EXTENDED_BEST     0.125 / 800
```

Başlatma ve günlük operasyon:

```bash
python3 scripts/run_goal_difference_shadow.py initialize \
  --initial-ratings path/to/2026_27_ao_first_elo.csv \
  --state-dir output/live_shadow_2026_27

python3 scripts/run_goal_difference_shadow.py lock \
  --state-dir output/live_shadow_2026_27 \
  --fixture path/to/fixture.csv

python3 scripts/run_goal_difference_shadow.py settle \
  --state-dir output/live_shadow_2026_27 \
  --result path/to/result.csv
```

Shadow state, ledger ve update dosyaları production checkpoint'inden ayrıdır.
En az 300 lig-aşaması+ maç ve 75 UCL maçı oluşunca zorunlu izleme incelemesi
yapılır.
Pooled forward sıralama farkı negatif olamaz; tek fold için izin verilen azami
gerileme Spearman'da `0.005`, pairwise'ta `0.0025`tir. Ara sonuçlara bakarak
alpha, tau veya cap değiştirilemez; güvenilir zarar görülürse geri alma kararı
yeni config fingerprint ile manuel verilir.
Repo'da henüz tüm 2026/27 katılımcılarını kapsayan başlangıç rating CSV'si
bulunmadığından bugün temiz prospective state başlatılamaz.

## UCL ve ClubElo Riski

Final `carry=0` ve standart 1X2 harici benchmarkında UCL sonucu:

```text
AO Dynamic 1X2 Brier       0.601615
AO Static 1X2 Brier        0.606032
ClubElo 1X2 Brier          0.572126
AO - ClubElo              +0.029489
Dependency envelope       [-0.002932, +0.064568]
```

AO Dynamic, AO Static'e karşı `-0.004417` iyileşir. ClubElo nokta tahmini daha
iyi olsa da dependency envelope sıfırı kestiği için fark final değerlendirmede
güvenilir değildir. Bu sonuç eşitlik veya sorunun çözüldüğü anlamına gelmez;
paired arşiv yalnız 171 unseen UCL maçı ve ağırlıkla yerleşik kulüpleri kapsar.
Risk 2026/27 holdout'ta UCL segmenti olarak ayrı raporlanacaktır.

## Frozen Contract ve Holdout

Makinece doğrulanan parametre ve kolon sözleşmesi:

```text
contracts/ao_european_elo_v2.json
```

2026/27 qualifying ve play-off maçları prospective holdout kapsamı dışındadır.
Temiz değerlendirme en erken 8 Eylül 2026'da başlayan lig aşaması maçlarından,
yalnızca kickoff öncesinde hash-zincirli loga yazılmış tahminlerle oluşur. Replay
çıktıları holdout'a dahil edilmez. Sonuçlara bakarak parametre ayarlanamaz, kapalı
katman açılamaz ve eski pre-match kayıtları yeniden yazılamaz. 2027/28 bir sonraki
tam sezon holdout adayıdır.

## Dokümantasyon

```text
docs/AO_EUROPEAN_ELO_V2_MODEL_CONTRACT.md
docs/HOLDOUT_PROTOCOL_2026_27.md
MODEL_STATUS.md
output/v2_evaluation_upgrade_2018_2026/evaluation_report.md
output/final_robustness_2018_2026/robustness_report.md
output/controlled_goal_progression_backtest_2018_2026/backtest_report.md
output/goal_shadow_parameter_search_2018_2026/parameter_search_report.md
output/pdf/AkilOyunu_Elo_Model_Aciklayici.pdf
output/pdf/AkilOyunu_Elo_Model_Kisa.pdf
output/pdf/AkilOyunu_VeriAnlamlandirma.pdf
```

İlk dosya tam teknik sözleşme, ikinci dosya güncel karar özeti, PDF'ler ise
detaylı model anlatımı, toplantı için kısa anlatım ve veri alan sözlüğüdür.
