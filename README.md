# AO European Elo

AO European Elo, UEFA kulüp turnuvaları için sezon öncesi başlangıç rating'i,
exact-UTC maç olaylarıyla güncellenen canlı Power Elo ve standart H/D/A maç
olasılıkları üreten açıklanabilir bir Python/Pandas motorudur. Kullanıcıya
sunulan güncel 1X2 tahmini, AO rating çekirdeğinin üzerine kurulu Current ML ve
AO Domestic Poisson bileşenlerini log-probability uzayında `%50/%50`
birleştirir.

Aktif geliştirme sözleşmesi `ao-european-elo-v2.0-dev-freeze` sürümüdür.
Parametreler 2018/19-2025/26 geliştirme verisinde nested walk-forward ile
seçilmiştir. 2026/27 elemeleri freeze tarihinden önce başladığı için sezonun
tamamı untouched sayılamaz. Prospective holdout, yalnızca sonuçsuz fixture ile
maçtan önce kilitlenen 2026/27 lig aşaması ve sonrası kayıtlarından oluşacaktır.
Bu nedenle model henüz tamamen kanıtlanmış production modeli olarak tanımlanmaz.
Prediction katmanı `PROMOTE_WITH_MONITORING` olarak operasyonel biçimde aktiftir;
bu ifade 2026/27 prospective doğrulamasının tamamlandığı anlamına gelmez.

Güncel production değerlendirmesinin ana belgesi:

```text
reports/current_model/current_model_evaluation_report.md
```

`output/` yeniden üretilebilir yerel çalışma alanıdır ve Git'e eklenmez.

## Model Özeti

Model dört değeri ayrı muhasebeleştirir:

```text
AO First Elo            sezon öncesi statik başlangıç gücü
Power Elo               maçlarla sıfır toplamlı değişen güç
Achievement Reserve     tur/kupa başarısı için ayrı rezerv
Progression Bonus       tamamlanan eleme turu için sezonluk sabit bonus

AO Live Elo = Power Elo + Achievement Reserve + Progression Bonus
```

Pre-match olasılık sunumu rating muhasebesinden ayrıdır:

```text
P_current_ml = LogBlend(P_AO, P_structural_ml, 0.90)
P_ao_poisson = LogBlend(P_AO, P_domestic_poisson_rho0, 0.50)
P_served     = LogBlend(P_current_ml, P_ao_poisson, 0.50)
```

Bu tahmin AO Live Elo'yu değiştirmez. Herhangi bir artifact, feature veya
Domestic Poisson state problemi satır bazında `P_AO` fallback'i üretir.

Aktif sürümde Achievement Reserve sıfırdır. Progression Bonus ise UCL/UEL/UECL
için `12/8/4` olarak aktiftir; bu nedenle eleme aşamalarında AO Live Elo,
Power Elo'dan sezonluk bonus kadar yüksek olabilir.

Takım exposure, geçmiş maç sayısı ve inactivity sinyallerinden üretilen Dynamic
K adayı da nested walk-forward kapısını geçmemiştir. Canlı güncelleme sabit
`K=103.9809863339` kullanır.

### AO First Elo v2

```text
M = 1500 / (903.92 - 500) = 3.713606654783126
AO Elo v2 = 500 + M x (AO Elo v1.1 - 500)
```

`500-2000` bir referans bandıdır ve hesap sonunda clipping uygulanmaz. Aktif
beta değerleri `0/0/0` iken Domestic Surprise öncesi bounded statik çekirdeğin
ulaşılabilir teorik üst sınırı `2000`, `+30` sürpriz düzeltmesiyle aktif AO First
Elo yapısal üst değeri `2030`dur. Dynamic Power/Live Elo kırpılmaz ve bu değeri
aşabilir. Sürpriz kapalıyken affine dönüşüm v1.1 takım sırasını korur. Dynamic Scale, saha
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
    Adjusted Domestic Prior
    + Effective European Exposure
      x (European Prior - Adjusted Domestic Prior)
```

Aktif modelde Domestic Prior, takımın son beş lig bitirişine göre varyans
kontrollü bir sürpriz düzeltmesinden geçer:

```text
Historical Mean = 0.07 x P(t-5) + 0.13 x P(t-4) + 0.20 x P(t-3)
                + 0.27 x P(t-2) + 0.33 x P(t-1)

Normalized Volatility = min(1, 2 x Weighted Volatility)
Consistency = 1 - 0.50 x Normalized Volatility
Effective Surprise = (P_current - Historical Mean) x Consistency

Domestic Surprise Adjustment = clip(
    594.1770647653 x Achievement Scale x 0.40 x Effective Surprise,
    -30,
    +30
)

Adjusted Domestic Prior = Domestic Prior + Domestic Surprise Adjustment
AO Surprise Effect = (1 - Effective European Exposure)
                   x Domestic Surprise Adjustment
```

`P=(league_team_count-position)/(league_team_count-1)` doğrudan lig
percentile'ıdır. Beş tam tarihsel sezon yoksa düzeltme sıfırdır. Varyans
sürprizin yönünü değiştirmez; yalnız geçmiş sıralaması oynak takımlarda etkiyi
azaltır.

Ham exposure oynanan sezon ve maç miktarından hesaplanır. Aktif effective
exposure tavanı `0.85`tir; tam Avrupa kanıtında Domestic Prior'ın yüzde 15'i
korunur. Avrupa geçmişi olmayan açık sıfır satırında final rating doğrudan
Adjusted Domestic Prior'dır.

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
Penaltı atışı golleri skora eklenmez. Shootout ile karara bağlanan tek veya çift
ayaklı eşleşmenin karar maçında Power Elo, gerçek saha sonucunu kullanır. Saha
galibiyeti `S=1`, beraberlik `S=0.5`, mağlubiyet `S=0` olur; shootout nedeniyle
gol farkı ve xG ek bonusları kapatılır ve gol çarpanı `1` kalır.

### European Progression Bonus

Tamamlanan eleme eşleşmesinin kazananına maç güncellemesinden sonra sabit bonus
eklenir:

```text
UCL  = +12 / etap, sezonluk cap 48
UEL  =  +8 / etap, sezonluk cap 32
UECL =  +4 / etap, sezonluk cap 16

AO Live Elo = Power Elo + Achievement Reserve + Progression Bonus
```

Uygun etaplar son 16, çeyrek final, yarı final ve finaldir. Knockout play-off
rota asimetrisi nedeniyle uygun değildir.
Bonus yalnız eşleşme tamamen kesinleşince ve aynı `tie_id` için bir kez verilir.
Kaybedenden ayrıca puan düşülmez; lig aşaması ve ön elemeler bonus üretmez.
Penaltıyla tur atlayan takım maç Elo'sunda beraberlik sonucunu korurken tur
bonusunu alır. Üç turnuva bakiyesi ayrı tutulur ve yeni sezonda sıfırlanır.

### Standart 1X2 Olasılıkları

`E_home` bir galibiyet olasılığı değil, normalize beklenen maç puanıdır. Aktif
olasılık katmanı bu değeri gerçek üç sınıflı çıktıya çevirir:

```text
P_draw = 0.24 x (4 x E_home x (1-E_home)) ^ 1.00
P_home = E_home - 0.5 x P_draw
P_away = 1 - E_home - 0.5 x P_draw
```

Tek maçta tamamlanan eleme eşleşmelerinde saha skoru 120 dakika sonunda da
berabere kalabildiği için ayrı ve sonuçtan bağımsız format girdisi kullanılır:

```text
normal / iki ayaklı maç draw_at_even = 0.24
tek maçlık eleme draw_at_even         = 0.12
```

`is_single_match_tie` yalnız fikstür formatından gelir; skor, uzatma sonucu,
penaltı veya tur atlayan takım kullanılarak türetilemez. Bu düzeltme Power Elo
ve `E_home` değerini değiştirmez, yalnız H/D/A olasılık ayrışmasını düzeltir.

Bu dönüşümde olasılıklar toplamı `1` ve
`P_home + 0.5 x P_draw = E_home` eşitliği zorunludur. Standart Brier, H/D/A
sınıflarındaki üç kare hatanın toplamıdır; standart log-loss gerçekleşen sınıfa
atanan olasılığın negatif logaritmasıdır.

## Aktif ve Kapalı Katmanlar

| Katman | Karar | Aktif değer |
| --- | --- | ---: |
| Country/European/Exposure tail | `NO_PROMOTION` | beta `0/0/0` |
| Domestic Surprise | `PROMOTE_MANUAL` | theta `0.40`, gamma `0.50`, cap `30` |
| Dynamic Scale/H/K | `PROMOTE` | `835.561/148.544/103.981` |
| Season carry | `DISABLE` | `0.00` |
| Standart 1X2 çıktı | `PROMOTE` | draw `0.24`, shape `1.00` |
| Tek maç format düzeltmesi | `ACTIVATE_STRUCTURAL` | draw `0.12`; rating state değişmez |
| Alt-1 beraberlik shape | `KEEP_SHADOW` | full fit `0.84`; production `1.00` |
| Kontrollü gol farkı | `PROMOTE_MANUAL` | `alpha=0.15`, `tau=300`, GD cap `4` |
| Bounded xG performansı | `PROMOTE_MANUAL` | ratio `0.30`, scale `1.25`, analitik minimum kazanım oranı `0.70` |
| Sıfır-toplamlı tur bonusu | `REJECT` | aktif base bonus `0` |
| Sabit kazanan-only progression | `PROMOTE_MANUAL` | R16 ve sonrası `12/8/4`, cap `48/32/16` |
| Achievement Reserve | `DISABLE` | base `0` |
| Normal maç turnuva K çarpanı | `DISABLE` | uygulanmaz |
| ML + Domestic Poisson 1X2 | `PROMOTE_WITH_MONITORING` | `%50/%50` log blend, `rho=0`, AO fallback, rating feedback yok |

Final robustness'taki eski LOG/SQRT gol farkı ailesi `3/6` Brier ve `2/5`
forward-ranking sonucu nedeniyle kapalı kalmıştır. Onun yerine 23 Temmuz
kontrollü deneyinde `1 + alpha x ln(min(GD,4)) x exp(-abs(D)/tau)` formülü
test edilmiştir. Önceden belirlenen `0.10/300` adayı kontrollü gol farkı
ailesinin ilk güçlü kanıtını üretmiştir. Sonraki bounded xG çalışması ve tam
sezon replay sonrasında manuel production kararı `alpha=0.15`, xG ratio `0.30`,
scale `1.25` ve analitik minimum kazanım oranı `0.70` olarak dondurulmuştur. Bu oran
ayrı bir runtime clamp değildir: `max_xg_ratio=0.30` olduğu için bounded xG formülü
zaten kazananın klasik sonuç residual'ını en fazla yüzde 30 azaltabilir. Sabit `12/8/4`
progression manuel ürün kararıyla aktiftir;
sıfır-toplamlı tur bonusu, turnuva K ve Achievement Reserve kapalıdır.

Beraberlik eğrisinde `shape < 1` artık geçerli bir araştırma alanıdır. Ham
beraberlik olasılığı negatif H/A olasılığı üretmemesi için
`min(raw_draw, 2*min(E,1-E))` zarfıyla korunur. Full-history optimumu `0.84`
olsa da training-only walk-forward sonuç Brier `4/6`, log-loss `3/6` ve pooled
losslarda çok küçük gerileme ürettiği için production shape `1.00` kalır.
Tek maç format düzeltmesinin tam örneklem optimumu `0.1129`dur. Sabit `0.12`,
4.884 değerlendirme maçında Brier'ı `0.000658`, log-loss'u `0.001617`
iyileştirmiştir. 248 tek maçın 200'ü 2020/21 sezonunda olduğu için bu sonuç
untouched terfi kanıtı sayılmaz; açık format hatasını düzelten production
sözleşmesi ve ayrıca izlenen bir duyarlılık sonucu olarak tutulur.

Aktif runtime manifesti:

```text
contracts/ao_european_elo_v2_production.json
```

Aktif prediction artifact manifesti:

```text
artifacts/production_prediction/manifest.json
```

Development-window ensemble sonucu `4.884` unseen maçta Brier `0.568093`,
log-loss `0.959242`, accuracy `0.553849` olmuştur. AO rating çekirdeği aynı
pencerede `0.572093/0.964371/0.550164` üretmiştir. Tarihsel otomatik gate
`KEEP_SHADOW` vermiş, kullanıcı onayıyla kontrollü fallback ve monitoring
şartları altında operasyonel karar `PROMOTE_WITH_MONITORING` olmuştur.

### Aktif Nihai Model

2025/26 tam sezon replay ve manuel ürün kararı sonrasında seçilen birleşim ayrı
bir sözleşmede dondurulmuştur:

```text
contracts/ao_european_elo_v2_final_candidate.json

candidate_status     = FINAL_MODEL_CANDIDATE
goal_alpha           = 0.15
goal_tau             = 300
goal_cap             = 4
xG ratio             = 0.30
xG scale             = 1.25
minimum winner ratio = 0.70
progression bonus    = R16 ve sonrası 12/8/4; cap 48/32/16
```

Aday maç güncellemesi:

```text
Delta_base = K x (S-E)
Delta_GD = Delta_base x (M_GD-1)
Q_xG = tanh((xG_home-xG_away) / 1.25)
Delta_xG = 0.30 x abs(Delta_base) x Q_xG
Delta_final = Delta_base + Delta_GD + Delta_xG
```

İki tarafın xG'si birlikte yoksa imputation yapılmaz ve maç `alpha=0.15` gol
farkı güncellemesine düşer. Beraberlikte ve penaltı atışlarıyla karara bağlanan
maçta xG düzeltmesi uygulanmaz. Penaltı kararı saha sonucunu değiştirmez: 90/120
dakika 2-0 bittiyse `S=1`, 1-1 bittiyse `S=0.5`; iki durumda da shootout
nedeniyle ek GD ve xG bonusları kapalıdır. Kazanan yönü, maç ve sezon toplam
Elo korunumu zorunludur. Runtime API'si:

```python
from ao_elo import load_final_candidate_runtime, update_final_candidate_match

runtime = load_final_candidate_runtime(
    "contracts/ao_european_elo_v2_final_candidate.json"
)
update = update_final_candidate_match(
    runtime,
    home_rating=1500,
    away_rating=1500,
    home_goals=2,
    away_goals=0,
    is_neutral=True,
    decided_on_penalties=False,
    xg_home=3.5,
    xg_away=0.5,
)
```

Bu statü production aktivasyonu değildir. Mevcut production contract geriye
dönük karşılaştırma ve prospective geçiş kontrolü için değiştirilmeden korunur.

## Proje Yapısı

```text
src/ao_elo/
  config.py           statik v1.1/v2 parametreleri
  features.py         statik feature hesapları
  domestic_surprise_variance.py beş sezonluk varyans kontrollü sürpriz
  scoring.py          prior, exposure ve final rating formülleri
  validators.py       statik CSV validation
  pipeline.py         AO First Elo dataframe/CSV motoru
  dynamic.py          reusable canlı Elo Python API'si
  dynamic_csv.py      CSV sözleşmeleri ve batch çekirdeği
  evaluation.py       adjusted ranking, 1X2 ve belirsizlik metrikleri
  robustness.py       azalan getirili gol ve joint competition-K adayları
  controlled_live.py  kontrollü GD ve sıfır-toplamlı progression çekirdeği
  dynamic_k.py        takım belirsizliği ve aday Dynamic K matematiği
  match_context.py    rövanş, saha, beraberlik ve sezon başlangıcı adayları
  progression_probability.py format-duyarlı tur geçme olasılığı
  scoreline.py         Elo-koşullu Poisson/Dixon-Coles shadow skor katmanı
  scoreline_calibration.py turnuva ve geçmiş-sezon gol seviyesi adayları
  final_candidate.py nihai alpha 0.15 + bounded xG sözleşme runtime'ı
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
  run_dynamic_k_backtest.py
  run_match_context_backtest.py
  run_domestic_regression_diagnostics.py
  run_rejected_feature_heterogeneity_audit.py
  run_progression_probability_calibration.py
  run_progression_reserve_final_backtest.py
  run_scoreline_backtest.py
  run_scoreline_level_calibration.py
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

Domestic Surprise için `domestic_context.csv` aşağıdaki opsiyonel kolon
çiftlerini kabul eder:

```text
history_position_t_minus_5, history_team_count_t_minus_5
history_position_t_minus_4, history_team_count_t_minus_4
history_position_t_minus_3, history_team_count_t_minus_3
history_position_t_minus_2, history_team_count_t_minus_2
history_position_t_minus_1, history_team_count_t_minus_1
```

Her sezon için iki değer birlikte dolu veya birlikte boş olmalıdır. Beş çiftin
tamamı geçerli değilse `domestic_surprise_status=INSUFFICIENT_HISTORY` üretilir
ve eski AO First Elo değiştirilmez.

Statik output'ta `ao_first_elo`, deterministik `ao_first_elo_rank`, uncapped ve
tail-adjusted normlar, saturation/tail bayrakları, Domestic/European prior ve
iki exposure alanı bulunur. Ayrıca tarihsel ortalama/varyans, consistency,
ham/efektif sürpriz, Domestic Prior düzeltmesi ve AO First Elo net etkisi ayrı
kolonlarda raporlanır.

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
xg_home, xg_away, xg_analysis_eligible,
is_neutral, decided_on_penalties
```

xG alanları opsiyoneldir ancak çift olarak gelmelidir. `xg_analysis_eligible`
false ise iki değer de boş bırakılır ve motor gol-farkı-only fallback kullanır.

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

## Dynamic K Backtesti

Takıma göre değişen K, aktif kontrollü gol farkı baseline'ına karşı 163 tekil
aday ve 6 nested outer fold ile test edilir:

```bash
python3 scripts/run_dynamic_k_backtest.py
```

Sonuç `KEEP_FIXED_K` olmuştur. Nested unseen karşılaştırmada Brier
`+0.00008489`, log-loss `+0.00011225` kötüleşmiş ve forward ranking yalnızca
`1/5` foldda güvenli kalmıştır. Düşük exposure bandında artan K belirgin zarar
üretmiştir. Production contract değiştirilmemiştir.

Tam çıktı:

```text
output/dynamic_k_backtest_2018_2026/backtest_report.md
output/dynamic_k_backtest_2018_2026/selected_dynamic_k_model.json
```

## Gol Farkı ve xG Ablation

xG katmanı AO First Elo'ya dokunmadan, yalnız maç sonrası Power Elo
güncellemesinde ayrı bir araştırma hattında test edilir:

```text
BASE    gol farkı kapalı, xG kapalı
GD      production 0.10 / 300 / cap 4
XG      gol farkı kapalı, xG harmanı aktif
GD_XG   production gol farkı ve xG harmanı birlikte
```

```text
S_xG = 1 / (1 + exp(-(xG_home - xG_away) / c_xG))
Delta = K x [(1-rho) x (S-E) x M_GD + rho x (S_xG-E)]
```

Eksik xG sıfırla doldurulmaz. XG kolu eksik maçta BASE'e, GD_XG kolu GD'ye
döner. Açık kaynak veri eşleştirmesi ve backtest şu komutlarla yeniden
çalıştırılır:

```bash
python3 scripts/build_xg_backtest_dataset.py \
  --fixtures path/to/fixtures.parquet \
  --match-stats path/to/match_stats.parquet \
  --leagues path/to/leagues.parquet \
  --teams path/to/teams.parquet

python3 scripts/run_xg_goal_ablation_backtest.py
```

Katı veri sözleşmesini 417 maç geçmiştir; 180 maçlık 2025/26 temporal
holdout'ta seçilen `GD_XG rho=0.05 / c_xG=1.00` adayı aktif GD'ye göre Brier
`-0.000680`, log-loss `-0.000950` iyileştirmiştir. Buna karşılık pooled
Spearman `-0.000540`, pairwise `-0.000257` gerilemiştir. Kullanılan xG
per-shot production metriği değil, şut bölgelerinden türetilmiş kaba ve
kesintili kapsamlı bir tahmindir. Bu nedenle xG `SHADOW_ONLY` kalır ve
production sözleşmesi değiştirilmez.

Tam çıktı:

```text
data/xg_backtest_2018_2026/README.md
output/xg_goal_ablation_backtest_2018_2026/backtest_report.md
output/xg_goal_ablation_backtest_2018_2026/selected_xg_model.json
```

## Tur Geçme Olasılığı ve Reserve Sonucu

Achievement Reserve'in nötr tek-maç `P_advance` varsayımı ayrıca
walk-forward olarak test edilir:

```bash
python3 scripts/run_progression_probability_calibration.py
python3 scripts/run_progression_reserve_final_backtest.py
```

1.606 tamamlanmış eşleşmede format-duyarlı olasılık kalibrasyonu identity
proxy'ye göre Brier'ı `-0.003531`, log-loss'u `-0.009814` iyileştirmiştir.
Bazı turnuva/format segmentleri gerilediği için kalibre olasılık production'a
alınmamış, reserve testinde güçlü bir shadow kontrolü olarak kullanılmıştır.

Reserve, güncel sabit K ve kontrollü gol farkı comparator'ına karşı Brier'da
`+0.004836`, log-loss'ta `+0.007107` kötüleşmiştir. Conservative Brier
aralığının tamamı zarar yönündedir; forward ranking guardrail'i `0/5`
olmuştur. Bu nedenle `Achievement Reserve = 0` kararı korunur. UCL/UEL/UECL
ayrımını ayrı reserve puanıyla canlı Elo'ya ekleme katmanı production'da
aktif değildir.

Tam çıktılar:

```text
output/progression_probability_2018_2026/calibration_report.md
output/progression_probability_2018_2026/selected_progression_probability.json
output/progression_reserve_final_backtest_2018_2026/backtest_report.md
output/progression_reserve_final_backtest_2018_2026/decision.json
```

## Maç Bağlamı Backtesti

Rövanş toplam skoru, dinamik saha avantajı, bağlamsal beraberlik ve
Domestic-Prior sezon regresyonu aynı current-production baseline'ına karşı
tek pakette test edilir:

```bash
python3 scripts/run_match_context_backtest.py
```

Veri 8 sezonda 6.340 maç, 2.106 eleme eşleşmesi ve 1.858 rövanş maçı içerir.
Her katman ayrı nested seçimden geçer; yalnız tek başına bütün kapıları geçen
özellik ortak modele taşınır.

Sonuçta toplam skor ve 108 dinamik saha profili her fold baseline'a dönmüştür.
Domestic regression için eski “her evaluable fold sıfırın altında olmayacak”
ranking vetosu kaldırılmıştır. Ranking farkları artık target-season cluster
bootstrap ve küçük örneklem t-aralığıyla ölçülür; yalnız conservative yüzde 95
aralığı tamamen zarar yönündeyse veto uygulanır.

Güncel kodla yeniden çalıştırılan bu tarihsel match-context baseline'ında
(`GD=0.10`, xG yok) Domestic regression ranking kapısını geçer:
ortalama Spearman farkı `+0.002560`, conservative aralık
`[-0.012265,+0.017384]` ve reliable harm `false` değerindedir. Bununla birlikte
Brier `-0.003339`, log-loss `-0.004885` olsa da Brier yalnız `3/6` fold kazanır
ve conservative Brier üst sınırı `+0.000110` ile sıfırı keser. Bu nedenle güncel
karar `NO_PROMOTION`dur. Production modeli ve `AO First Elo` sözleşmesi
değişmemiştir.

Tam çıktı:

```text
output/match_context_backtest_2018_2026/backtest_report.md
output/match_context_backtest_2018_2026/decision_manifest.json
```

İlk başarısız Domestic-Prior ranking foldu takım seviyesinde ayrıca incelenir:

```bash
python3 scripts/run_domestic_regression_diagnostics.py
```

2021/22'nin ilk UECL sezonundaki pooled kayıp küçüktür ve üç UECL gözlemine
duyarlıdır; bu gözlemlerden herhangi biri çıkarıldığında iki sıralama farkı
da pozitif olur. Bu bulgu artık tek başına ret sebebi değildir. Post-hoc
persistence taraması yalnız diagnostik kalır; production kararı complete
walk-forward loss ve ranking belirsizlik kapılarından verilir.

```text
output/domestic_regression_diagnostics_2018_2026/diagnostic_report.md
output/domestic_regression_diagnostics_2018_2026/diagnostic_manifest.json
```

Global retlerin birkaç takım outlier'ından kaynaklanıp kaynaklanmadığı ayrıca
denetlenir:

```bash
python3 scripts/run_rejected_feature_heterogeneity_audit.py
```

Achievement Reserve, Competition K ve bağlamsal beraberlikte zarar geniş ve
sistematiktir; hiçbir tek takım, sezon veya turnuva çıkarıldığında karar tersine
dönmez. Dynamic K bireysel takım değil UECL segmenti nedeniyle heterojendir.
Bu denetimdeki eski sıfır-toplamlı progression adayının çok küçük faydası
2025/26 ve UECL'ye bağımlıdır. Bu post-hoc segment bulguları, ondan ayrı olan
sabit kazanan-only `12/8/4` katmanının aktivasyon kanıtı olarak kullanılmaz.

```text
output/rejected_feature_heterogeneity_2018_2026/audit_report.md
output/rejected_feature_heterogeneity_2018_2026/audit_manifest.json
```

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

## Elo-Koşullu Skor Tahmin Backtesti

AO Live Elo'yu değiştirmeden skor dağılımı üreten bağımsız Poisson/Dixon-Coles
katmanı altı expanding outer fold ile test edilir:

```bash
python3 scripts/run_scoreline_backtest.py
```

Model maç öncesi AO Live Elo farkını `lambda_home` ve `lambda_away` gol
beklentilerine çevirir. Penaltı atış golleri hariç tutulur; 90 dakika veya
uzatma oynandıysa 120 dakika saha skoru hedeflenir. Skor matrisi en az 0-10
aralığındadır ve dışarıda kalan olasılık `1e-10` altına inene kadar adaptif
büyür. Gerçek OOS lambda değerleri nedeniyle güvenli üst destek 25 goldür.

4.884 unseen maçta exact-score NLL, ev/deplasman ortalamalarını ayrı öğrenen
intercept-only Poisson'u `6/6` fold geçti. Buna karşılık türetilen 1X2
olasılıkları mevcut AO 1X2 modelini yalnız `2/6` fold geçti; pooled Brier farkı
`+0.001388`, log-loss farkı `+0.002290` oldu. Full-data `rho=0.00` seçildi;
Dixon-Coles düşük skor düzeltmesi bağımsız Poisson'a ek fayda göstermedi ve
OOS score NLL'yi `+0.000403` kötüleştirdi.

Karar `KEEP_SHADOW`dur. Skor katmanı production sözleşmesine bağlanmamış ve
ratinglere geri beslenmemiştir. Ayrıntılı çıktılar:

```text
output/scoreline_backtest_2018_2026/backtest_report.md
output/scoreline_backtest_2018_2026/selected_scoreline_model.json
output/scoreline_backtest_2018_2026/unseen_predictions.csv
```

### Turnuva ve Geçmiş-Sezon Gol Seviyesi

Poisson katmanındaki toplam gol yanlılığı, ratinglere dokunmayan iki çarpanla
ayrıca test edilir:

```text
lambda_cal = lambda x exp(c_competition + c_prior_season)
```

Turnuva gücü ve geçmiş sezon gücü `0/0.25/0.50/0.75/1.00`, sezon penceresi
`1/2/3` tamamlanmış sezon olmak üzere 65 aday her outer fold içinde ayrı inner
walk-forward ile seçilir:

```bash
python3 scripts/run_scoreline_level_calibration.py
```

Nested seçim turnuva düzeltmesini hiçbir outer foldda seçmemiştir. Sezon
düzeltmesi dört foldda seçilmiş, fakat score NLL ve mevcut AO 1X2 karşısında
yalnız `2/6` fold kazanmıştır. Pooled farklar score NLL'de `+0.001527`,
Brier'da `+0.001086`, log-loss'ta `+0.001771` olmuştur. UCL gol bias değeri
`-0.202429`dan `-0.179161`e yaklaşsa da UEL score NLL'de güvenilir zarar
oluşmuştur.

Karar yine `KEEP_SHADOW`dur; production skor veya Elo sözleşmesi değişmez.

```text
output/scoreline_level_calibration_2018_2026/backtest_report.md
output/scoreline_level_calibration_2018_2026/selected_level_model.json
```

## Sabit Turnuva Bonusu

Tek görünür Elo yaklaşımını koruyan sabit ilerleme bonusu şu sözleşmeyle test
edilir:

```text
AO Live Elo = Power Elo + sezonluk turnuva bonusu
UCL / UEL / UECL etap bonusu = 12 / 8 / 4
UCL / UEL / UECL cap = 48 / 32 / 16
```

Yalnız tamamen sonuçlanmış son 16, çeyrek final, yarı final ve final
eşleşmesinin kazananı bonus alır. Knockout play-off bonus üretmez; böylece lig
aşamasını ilk 8'de bitirip play-off'u atlayan takıma karşı rota kaynaklı bir
dezavantaj oluşmaz. Kaybedenden puan düşülmez ve bonus yeni sezonda sıfırlanır.
Lig aşaması ile ön elemeler de bonus üretmez.

Altı foldun `4/6` tanesinde Brier ve log-loss iyileşmiş; pooled farklar sırasıyla
`-0.000006941` ve `-0.000011253` olmuştur. Etki küçük, güven aralıkları sıfırı
kesiyor ve UEL segmenti hafif geriliyor. Otomatik yüksek-bonus nested seçimi
**KEEP_DISABLED** kalmıştır. Önceden belirlenen konservatif `12/8/4` adayı ise
güvenilir zarar göstermemesi, kontrollü cap'leri ve açık kullanıcı onayıyla
`PROMOTE_MANUAL` olarak production'a alınmıştır. Bu karar istatistiksel kesinlik
iddiası taşımaz ve 2026/27 prospective kayıtlarında ayrıca izlenecektir.

```bash
python3 scripts/run_fixed_tournament_bonus_backtest.py
python3 scripts/verify_progression_bonus_production.py
```

```text
output/fixed_tournament_bonus_backtest_2018_2026/backtest_report.md
output/fixed_tournament_bonus_backtest_2018_2026/decision.json
output/progression_bonus_production_verification_2025_26/verification.json
```

### Aşama Ağırlıklı Bonus Deneyi

KPO bonusunu kaldıran ve toplam sezon bonusunu son 16, çeyrek final, yarı final
ve şampiyonluk arasında ağırlıklandıran profiller ayrıca nested walk-forward ile
test edilmiştir. Gentle, Linear ve Final Heavy profilleri UCL toplam cap
`30/45/60/75/90`; UEL ve UECL ise sabit `2/3` ve `1/3` oranlarıyla taranmıştır.

Nested seçim iki foldda `Gentle + cap 30`, dört foldda mevcut production
`12/8/4 × 5` modelini seçmiştir. Pooled Brier ve log-loss sırasıyla yalnız
`-0.000017673` ve `-0.000022539` iyileşmiş, fakat güven aralıkları sıfırı
kesmiştir. Aynı-sezon ranking iki metrikte birlikte `0/6`, forward-ranking ise
`0/5` iyileşme göstermiştir. Tam geçmiş seçimi de mevcut production olmuştur.
Bu tarihsel karar daha sonra rota asimetrisi denetimiyle geçersiz kılınmıştır:
aktif production sözleşmesi KPO'yu dışlayan dört aşamalı yapıdır.

KPO'yu koruyup toplam cap'i yine `60/40/20` bırakan beş aşamalı ağırlık testi
de ayrıca yapılmıştır. Gentle `%10/15/20/25/30`, Linear `1/2/3/4/5` oranlı ve
Late Heavy `%5/10/15/25/45` profilleri, mevcut Equal Five `%20` dağılımıyla
karşılaştırılmıştır. Altı foldun tamamında eğitim seçimi Equal Five production
profilinde kalmıştır. Üç ağırlıklı profil pooled Brier ve log-loss'u çok küçük
ölçüde kötüleştirmiş; üçünde de UECL log-loss için güvenilir zarar görülmüştür.
Tam geçmiş seçimi yine production profilidir.

```bash
python3 scripts/run_stage_weighted_progression_backtest.py
python3 scripts/run_five_stage_weighted_progression_backtest.py
```

```text
output/stage_weighted_progression_backtest_2018_2026/backtest_report.md
output/stage_weighted_progression_backtest_2018_2026/selected_candidate.json
output/five_stage_weighted_progression_backtest_2018_2026/backtest_report.md
output/five_stage_weighted_progression_backtest_2018_2026/selected_candidate.json
```

## 2025/26 Tam Sezon Replay

2025/26 sezonundaki 236 takım ve 961 UCL/UEL/UECL maçı kesin UTC sırasıyla
yeniden işlenir:

```bash
python3 scripts/run_2025_26_season_replay.py
```

Çalışma iki kanıt sınıfını bilinçli olarak ayırır. `OOS_FOLD6`, yalnız 2024/25
sonuna kadar seçilmiş fold-6 parametrelerini kullanır. Bugünkü production ve
shadow kolları `RETROSPECTIVE_COUNTERFACTUAL` olarak etiketlenir ve bağımsız
doğrulama sayılmaz. İki taraf için xG bulunan ortak 180 maç ise ana replay'den
ayrı `MATCHED_XG_APPENDIX` kapsamındadır.

Kilitli historical referansın 1X2 Brier/log-loss değerleri
`0.587906/0.987690` olmuştur. Önceden seçilmiş `6/4/2` bonus duyarlılığı
farkları `-0.000047/-0.000066` yönünde iyileştirmiş, ancak konservatif cluster
güven aralıkları sıfırı kesmiştir. Current production counterfactual sonucu
`0.586725/0.985323`tür. Gol farkını kapatmak iki metriği de hafif
kötüleştirmiştir; bu bulgu production gol farkı katmanıyla uyumludur fakat
retrospective olduğu için yeni terfi kanıtı değildir.

`12/8/4` fixed bonus ve historical `6/4/2` duyarlılığı
`CONSISTENT_SHADOW_SIGNAL`; daha güçlü gol farkı adayları ve domestic anchor
`MIXED_OR_INCONCLUSIVE`; gol farkını kapatma kolu `HARM_SIGNAL` olarak
raporlanmıştır. Hiçbir statü veya production parametresi değiştirilmemiştir.

Ana çıktılar:

```text
output/season_replay_2025_26/replay_report.md
output/season_replay_2025_26/model_comparison.csv
output/season_replay_2025_26/match_predictions_and_updates.csv
output/season_replay_2025_26/scoreline_predictions.csv
output/season_replay_2025_26/AO_2025_26_Sezon_Replay_Raporu.pdf
```

Varsayılan `4000` bootstrap kullanılır. Tüm sayısal CSV/JSON/Markdown çıktıları
aynı input ve manifest ile byte-stable üretilir. PDF, rapor verilerinden ayrıca
oluşturulur ve sayfa render kontrolünden geçirilir.

## 2025/26 UEFA Maç ve Ücretsiz xG Veri Seti

Tam 961 maçlık UEFA sonuç tablosu ile ücretsiz FotMob shot-based xG kapsamı şu
komutla yeniden üretilir:

```bash
python3 scripts/build_2025_26_xg_dataset.py
```

Ana CSV `281` UCL, `271` UEL ve `409` UECL maçı ile 236 AO takımını içerir.
FotMob xG bulunmayan maçlar tabloda kalır; ana xG kolonları boş bırakılır ve
eksiklik nedeni kontrollü bir kodla yazılır. API-Football tabanlı coarse xG
yalnız `secondary_xg_*` kolonlarında tutulur ve birincil xG'yi doldurmaz.

```text
data/xg_2025_26/uefa_2025_26_matches_with_xg.csv
data/xg_2025_26/xg_identity_audit.csv
data/xg_2025_26/xg_coverage_summary.csv
data/xg_2025_26/xg_source_comparison.csv
data/xg_2025_26/source_manifest.json
```

Kaynak yanıtları `_source_cache/` altında devam edilebilir biçimde saklanır ve
Git'e eklenmez. `--offline` aynı cache üzerinden ağ isteği olmadan deterministik
çıktı üretir. Bu veri seti analiz girdisidir; AO First Elo veya AO Live Elo
production sözleşmesini değiştirmez.

FotMob kapsamında 606 maç (`UCL=226`, `UEL=227`, `UECL=153`) analiz için
uygundur. Daha önce coarse xG üzerinde seçilmiş parametreleri 2025/26'da yeniden
seçmeden doğrulamak için:

```bash
python3 scripts/run_fotmob_xg_backtest_2025_26.py
```

Convex ve yön-korumalı `rho=0.05 / c_xG=1.00` kolları hem 961 maçlık akışta
hem ortak 606 maçta Brier ve log-loss'u küçük fakat clustered CI açısından
güvenilir biçimde iyileştirmiştir. Aynı sezon sıralama ölçümü ise sırasıyla
`-0.001929` Spearman ve `-0.000978` pairwise gerilemiştir. Daha güçlü additive
`rho=0.50 / c_xG=0.75` daha büyük loss iyileşmesi üretmiş, fakat güven aralığı
sıfırı kesmiş ve sıralama kaybı büyümüştür. Ranking-first kuralı nedeniyle tüm
kollar `MIXED_OR_INCONCLUSIVE` ve xG `SHADOW_ONLY` kalır.

```text
output/fotmob_xg_backtest_2025_26/backtest_report.md
output/fotmob_xg_backtest_2025_26/selected_shadow_model.json
```

Skor farkının kazanan takımın xG üstünlüğüyle desteklenmeyen kısmını azaltan
`UNSUPPORTED_MARGIN` ailesi ayrıca test edilir:

```text
U = max(0, min(abs(GD), 4) - winner_xg_advantage - tolerance)
P = max(minimum_multiplier, 1 - lambda * ln(1 + U))
Delta_final = Delta_GD * P
```

```bash
python3 scripts/run_unsupported_margin_backtest_2025_26.py
```

Grid `1 Ocak 2026` öncesindeki 399 xG maçında seçilmiş, sonraki tamamı xG
kapsamlı 207 maçta doğrulanmıştır. Development seçimi
`tolerance=1.00`, `lambda=0.20`, `minimum_multiplier=0.70` olmuştur; ancak
36 adayın hiçbiri development ranking guardrail'ini geçmemiştir. Seçilen aday
validation'da Brier `+0.000427`, log-loss `+0.000280`, Spearman `-0.003238`
ve pairwise `-0.001698` üretmiştir. UECL ve UEL iyileşirken UCL gerilemiştir.
Sonuç `MIXED_OR_INCONCLUSIVE_SHADOW`; production değişmemiştir.

```text
output/unsupported_margin_backtest_2025_26/backtest_report.md
output/unsupported_margin_backtest_2025_26/selected_unsupported_margin_model.json
```

Amaç xG'nin temel maç sonucu Elo'suna değil yalnız gol farkı bonusuna güvenlik
katmanı olmasıysa doğru sözleşme şudur:

```text
Delta_base = K x (S-E)
Delta_bonus = K x (S-E) x (M_GD-1)
Delta_final = Delta_base + xG_guard x Delta_bonus
```

```bash
python3 scripts/run_xg_goal_bonus_guard_backtest_2025_26.py
```

Development `tolerance=1.00`, `lambda=0.10`, bonus tabanı `0.75` adayını
seçmiştir. Validation'da 56 maçın yalnız gol bonusu ortalama `%6.7` azaltılmış;
doğruluk değişmemiş, pairwise `+0.000103`, ECE iyileşmiş, fakat Brier
`+0.000027`, log-loss `+0.000029` ve Spearman `-0.000290` olmuştur. Farklar
cluster güven aralığında güvenilir zarar değildir. Tek sezon kanıtı ve katı
ranking-first kapısı nedeniyle sonuç `MIXED_OR_INCONCLUSIVE_SHADOW` olarak
kalır; bu guard kolunda temel Elo xG ile azaltılmaz.

```text
output/xg_goal_bonus_guard_backtest_2025_26/backtest_report.md
output/xg_goal_bonus_guard_backtest_2025_26/selected_goal_bonus_guard_model.json
```

Kazanan takımın temel Elo kazancını koruyup xG farkını bağımsız, çift yönlü
performans bonusu olarak kullanan nihai araştırma sözleşmesi de walk-forward
olarak test edilir:

```text
Delta_base = K x (S-E)
Delta_GD = Delta_base x (M_GD-1)
Q_xG = tanh((xG_home-xG_away) / xG_scale)
Delta_xG = beta x abs(Delta_base) x Q_xG
Delta_raw = Delta_base + Delta_GD + Delta_xG
Delta_final = kazanan yönünü koruyan pozitif taban(Delta_raw)
```

```bash
python3 scripts/run_xg_performance_bonus_walk_forward_2025_26.py
```

Toplam 201 kol, tamamı xG kapsamlı ve birbirinden ayrık beş test döneminde
değerlendirilmiştir. Her fold katsayıyı yalnız önceki maçlardan seçmiştir. xG
adayı beş foldun tamamında seçilmiş; 387 unseen maçta Brier `-0.005023`,
log-loss `-0.007137`, pooled sıralama `+0.052620` ve pairwise `+0.018818`
yönünde iyileşmiştir. UCL ve UEL iyileşirken UECL Brier `+0.003886`, log-loss
`+0.003691` gerilemiş; cluster güven aralıkları da sıfırı kesmiştir. Tam sezon
verisinden yalnız gelecek shadow izleme için seçilen katsayılar `beta=1.50`,
`xG_scale=3.00`, minimum kazanan kazancı `%5`tir. Kazanan yönü ve sıfır toplam
invariantları korunur; karar `NO_PROMOTION_KEEP_PRODUCTION`dır.

```text
output/xg_performance_bonus_walk_forward_2025_26/backtest_report.md
output/xg_performance_bonus_walk_forward_2025_26/selected_xg_performance_bonus_model.json
```

Agresif `%5` tabanlı adayın pilotta temel kazancı neredeyse silebildiği
görüldükten sonra production için daha kontrollü bir aile ayrıca sınanır:

```text
Delta_base = K x (S-E)
Delta_GD_bonus = Delta_base x (M_GD-1)
Q_xG = tanh((xG_home-xG_away) / xG_scale)
Delta_xG = max_xg_ratio x abs(Delta_base) x Q_xG
Delta_final = Delta_base + Delta_GD_bonus + Delta_xG
```

```bash
python3 scripts/run_bounded_xg_adjustment_walk_forward_2025_26.py
```

`max_xg_ratio` yalnız `0.15/0.20/0.25/0.30` değerlerini alır; bu nedenle xG
temel maç sonucu kazancını en fazla `%30` azaltabilir veya artırabilir. Yirmi
sekiz xG adayı ve production kontrolü beş fold üzerinde Brier-first, ranking
guardrail'li seçimle test edilmiştir. Gelecek prospective shadow için tam sezon
seçimi `max_xg_ratio=0.30`, `xG_scale=1.25` olmuştur. Sabit adayın 387 unseen
maçtaki Brier değeri `0.591137 → 0.588595`, log-loss değeri
`0.990346 → 0.986518`, doğruluğu `%52.71 → %53.75` olmuştur. UCL, UEL ve
UECL'nin tamamında iki loss metriği de iyileşmiştir. Cluster CI sıfırı çok az
kestiği için production değişmemiş, aday prospective doğrulamaya bırakılmıştır.

```text
output/bounded_xg_adjustment_walk_forward_2025_26/backtest_report.md
output/bounded_xg_adjustment_walk_forward_2025_26/selected_bounded_xg_model.json
```

Seçilen `0.30/1.25` shadow katsayısı ayrıca 21 kontrollü pilot senaryoda ve
2025/26'nın 961 maçlık tam sezon replay'inde production ile yan yana
çalıştırılmıştır:

```bash
python3 scripts/run_bounded_xg_adjustment_pilot.py
python3 scripts/run_bounded_xg_full_season_replay_2025_26.py
```

Dengeli nötr 1-0 örneğinde klasik `51.99` Elo; xG `2.5-0.5` olduğunda `66.37`,
xG `0.2-2.0` olduğunda `38.05`, uç `0.1-4.5` olduğunda `36.42` olur. Tam
sezonda xG yalnız 606 doğrulanmış maçta aktif, kalan 355 maçta düzeltme
sıfırdır. Tüm 961 maçta Brier `0.589573 → 0.588284`, log-loss
`0.989154 → 0.987275`, doğruluk `%53.07 → %53.49` olmuştur. Maç başına
ortalama mutlak xG düzeltmesi `6.19`, en uç kazanan düzeltmeleri
`+24.86/-23.10` Elo'dur. Sezon sonu production farkının medyanı `0.44`, yüzde
90 değeri `31.10`, maksimumu `79.02` Elo'dur. UCL ve UEL iyileşmiş, tam sezon
UECL farkı çok küçük zarar yönündedir; replay retrospective olduğu için statü
değiştirmez.

```text
output/bounded_xg_adjustment_pilot/pilot_report.md
output/bounded_xg_full_season_replay_2025_26/replay_report.md
output/bounded_xg_full_season_replay_2025_26/final_ratings_comparison.csv
```

Gol farkı etkisini kontrollü xG katmanından bağımsız olarak görünür kılmak için
yalnız `goal_alpha` ayrıca test edilmiştir. Bu çalışmada xG
`ratio=0.30/scale=1.25`, `goal_tau=300` ve `goal_cap=4` sabit tutulmuş;
`goal_alpha=0.10/0.125/0.15/0.175/0.20/0.225/0.25` adayları beş expanding
walk-forward foldda karşılaştırılmıştır. Maç hareketine ayrıca `80 Elo` veya
başka bir üst sınır uygulanmamıştır.

Nested seçimin 387 unseen maçtaki Brier farkı `+0.000136`, log-loss farkı
`+0.000202` ve ranking farkı `-0.002809` olmuştur. Bu nedenle production
`goal_alpha=0.10` korunur. Tüm unseen sonuçlara sonradan bakıldığında `0.25`
en düşük Brier'ı üretse de UECL Brier'ı `+0.000585` kötüleşmiş ve clustered
güven aralığı sıfırı kesmiştir; bu değer terfi kanıtı değildir. Yalnız geçmiş
veriden yapılan tam-veri seçimi `0.15`i 2026/27 prospective shadow adayı olarak
işaretlemiştir; production sözleşmesini değiştirmez.

```text
output/goal_alpha_with_bounded_xg_backtest_2025_26/backtest_report.md
output/goal_alpha_with_bounded_xg_backtest_2025_26/selected_goal_alpha_model.json
output/goal_alpha_with_bounded_xg_backtest_2025_26/fixed_candidate_competition_summary.csv
```

`goal_alpha=0.20` ile kontrollü xG'nin birlikte davranışı ayrıca 21 sentetik
senaryoda gösterilebilir:

```bash
python3 scripts/run_goal_alpha_020_xg_pilot.py
```

Bu pilot klasik Elo, mevcut `alpha=0.10 + xG`, yalnız `alpha=0.20` gol farkı
ve `alpha=0.20 + xG` sonuçlarını aynı satırda karşılaştırır. Dengeli 2-0
senaryosunda alpha artışı toplam kazanca `3.60`, 4-0 senaryosunda `7.21` Elo
ekler. Tek farklı sonuçta gol bonusu bulunmadığından alpha fark yaratmaz; xG
düzeltmesi yine temel sonucu en fazla yüzde 30 oranında destekler veya azaltır.
Bu çıktı davranış testi olup production terfi kanıtı değildir.

```text
output/goal_alpha_020_xg_pilot/pilot_report.md
output/goal_alpha_020_xg_pilot/pilot_results.csv
```

Seçilen dengeli aday `goal_alpha=0.15`, kontrollü xG ile birlikte 2025/26'nın
961 maçında üç kollu tam sezon replay'e de alınmıştır:

```bash
python3 scripts/run_goal_alpha_015_xg_full_season_replay_2025_26.py
```

Kollar `0.10/xG kapalı`, `0.10+xG` ve `0.15+xG`dir. `0.15+xG`, `0.10+xG`ye
göre tüm maçlarda Brier'ı yalnız `-0.000002` değiştirirken log-loss
`+0.000010`, accuracy `-0.001041` olmuştur; sonuçlar pratik olarak eşittir.
Schedule-adjusted ranking `+0.001941`, pairwise accuracy `+0.000720`
iyileşmiştir. Maç başına ortalama mutlak ek alpha etkisi `0.47`, maksimumu
`3.65`; sezon sonu medyan mutlak takım farkı `1.03`, maksimum fark `7.81`
Elo'dur. UCL/UEL loss yönü olumlu, UECL küçük zarar yönündedir ve tüm cluster
CI'ları sıfırı keser. Replay adayın kontrollü olduğunu gösterir; retrospective
olduğu için tek başına production terfi kanıtı değildir.

```text
output/goal_alpha_015_xg_full_season_replay_2025_26/replay_report.md
output/goal_alpha_015_xg_full_season_replay_2025_26/model_comparison.csv
output/goal_alpha_015_xg_full_season_replay_2025_26/final_ratings_comparison.csv
```

## Takım Bazlı Home/Away Context Shadow

Takımın iç saha davranışı ile deplasman direnci, global ev sahibi avantajını
kalıcı ratinge eklemeden prediction-only bir katmanda test edilmiştir. Her takım
için yalnız önceki tamamlanmış sezonların pre-match residual'ları kullanılır;
yeni veya az verili takım `H=148.544266` global prior'ına shrink olur. Nötr
sahada takım etkileri uygulanmaz ve AO Live Elo state'i değişmez.

```text
H_effective = clip(148.544266 + HomeEffect_home - AwayEffect_away, 0, 300)
```

96 adaylı altı-fold nested backtestte Brier ve log-loss birlikte `5/6` foldda
iyileşmiştir. Pooled farklar sırasıyla `-0.000885` ve `-0.001265`tir; UCL,
UEL ve UECL point estimate'larının hiçbiri zarar yönünde değildir. Ancak pooled
dependency CI sıfırı az miktarda kestiği ve context sınırı sık kullanıldığı için
karar `KEEP_SHADOW_CANDIDATE`dır. Production global `H` ve Elo güncelleme
sözleşmesi değişmemiştir.

```bash
python3 scripts/run_team_venue_context_backtest.py --bootstrap-samples 4000
```

```text
output/team_venue_context_backtest_2018_2026/backtest_report.md
output/team_venue_context_backtest_2018_2026/selected_candidate.json
output/team_venue_context_backtest_2018_2026/profile_summary.csv
```

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
output/scoreline_backtest_2018_2026/backtest_report.md
output/scoreline_level_calibration_2018_2026/backtest_report.md
output/fixed_tournament_bonus_backtest_2018_2026/backtest_report.md
output/season_replay_2025_26/replay_report.md
output/season_replay_2025_26/AO_2025_26_Sezon_Replay_Raporu.pdf
docs/pdf/AkilOyunu_Elo_Model_Aciklayici.pdf
docs/pdf/AkilOyunu_Elo_Model_Kisa.pdf
docs/pdf/AkilOyunu_VeriAnlamlandirma.pdf
reports/current_model/current_model_evaluation_report.md
```

İlk dosya tam teknik sözleşme, ikinci dosya güncel karar özeti, PDF'ler ise
detaylı model anlatımı, toplantı için kısa anlatım ve veri alan sözlüğüdür.
