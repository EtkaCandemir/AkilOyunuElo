# AO European Elo

AO European Elo, UEFA kulüp turnuvaları için sezon öncesi başlangıç rating'i üreten bir Python/Pandas hesap motorudur. Model; ülke/lig gücü, yerel lig/kupa başarısı ve kulübün son yıllardaki Avrupa performansını tek bir başlangıç puanında birleştirir.

Bu proje aktif v1.1 statik başlangıç modelini ve sıralama öncelikli tarihsel
kalibrasyon backtestlerini içerir. `20 / 2.0 / 360` ülke adayı aggregate sonucu
iyileştirse de sıfır-exposure gerçekçilik kontrolünü geçemediği için production'a
alınmamıştır. Maç sonrası dinamik Elo güncellemesi bu modelden ayrı tutulur.

## Ana Formül

```text
AO First Elo =
Domestic Prior
+ Effective European Exposure * (European Prior - Domestic Prior)

Effective European Exposure = min(European Exposure, 0.85)
```

- `Domestic Prior`: Ülke/lig gücü ve yerel başarı sinyali.
- `European Prior`: Kulübün son 5 sezondaki Avrupa performansı.
- `European Exposure`: Avrupa performansı sinyalinin kaç sezon/maçlık veriye dayandığını gösteren ham kanıt ölçüsü.
- `Effective European Exposure`: Final rating karışımında kullanılan ve `0.85` ile sınırlandırılan etki ölçüsü.

Avrupa geçmişi olmayan takımlar cezalandırılmaz. Bu durumda iki exposure değeri de `0` olur ve final puan `Domestic Prior` değerine eşit kalır. Ham exposure `1.0` olsa bile final rating en az `%15` Domestic Prior katkısı taşır.

## Proje Yapısı

```text
src/ao_elo/
  config.py        # Model parametreleri
  features.py      # Feature hesapları
  scoring.py       # Rating formülleri
  validators.py    # Input validation kuralları
  pipeline.py      # CSV/dataframe hesap pipeline'ı

tests/
  test_achievement.py
  test_exposure.py
  test_scoring.py

data/pilot_10_teams/
  teams.csv
  country_coefficients.csv
  domestic_context.csv
  club_european_points.csv

scripts/
  run_pilot_10_teams.py
  build_pilot_inspection_report.py
  build_data_requirements_pdf.py

output/
  pilot_10_teams/
  pdf/
```

## Gerekli CSV Dosyaları

Model dört ana input CSV bekler:

1. `teams.csv`
   - Takım kimliği, ülke, ülke kodu ve yerel lig bilgisi.

2. `country_coefficients.csv`
   - Son 5 sezon UEFA ülke katsayı puanları.
   - Lig/ülke gücü hesabında kullanılır.

3. `domestic_context.csv`
   - Lig pozisyonu, ligdeki takım sayısı, lig şampiyonluğu, kupa şampiyonluğu ve Avrupa giriş bilgisi.
   - Yerel başarı skorunu üretir.

4. `club_european_points.csv`
   - Kulübün son 5 sezon Avrupa puanları, Avrupa oynayıp oynamadığı, maç sayıları ve match cap değerleri.
   - European Prior ve European Exposure hesabında kullanılır.
   - Her hedef takım için bir satır zorunludur. Avrupa geçmişi yoksa beş sezonun puan, played ve matches değerleri açıkça `0` yazılır.

### Sezon Tanımı

CSV dosyalarındaki `season`, rating'in üretildiği hedef sezonu gösterir. `t`, hedef sezon başlamadan önce tamamlanmış en güncel sezon; `t_minus_4` ise beş sezonluk pencerenin en eski sezonudur. Bu kural ülke ve kulüp puanlarına aynı şekilde uygulanır ve geleceğe ait veri kullanımını engeller.

### Veri Güvenliği

- `teams.csv`: `team_id` tekil olmalıdır.
- `country_coefficients.csv`: `season + country_code` tekil olmalıdır.
- `domestic_context.csv`: tek bir hedef sezon içermeli ve `season + team_id` tekil olmalıdır.
- `club_european_points.csv`: `season + team_id + country_code` tekil ve tüm hedef takımlar için mevcut olmalıdır.
- Eksik, negatif, sonsuz veya sayısal olmayan ülke/kulüp puanları reddedilir.
- Boolean alanlar yalnızca `true/false` veya `0/1` kabul eder.
- `official_five_year_total`, `official_country_rank`, `official_club_coefficient` ve `country_part` opsiyonel denetim alanlarıdır; ana formülde kullanılmaz.

Output CSV, ara model metriklerine ek olarak hedef `season`, `domestic_league`,
`domestic_position`, `league_team_count`, ham `european_exposure` ve rating'de
kullanılan `effective_european_exposure` alanlarını içerir. Final rating kolonunun
kalıcı adı `ao_first_elo`, sıralama kolonu `ao_first_elo_rank` değeridir. Satırlar
rating azalan, eşitlikte `team_id` artan sırada yazılır.

Detaylı alan sözlüğü için:

```text
output/pdf/AkilOyunu_VeriAnlamlandirma.pdf
```

## Kurulum

```bash
python3 -m pip install -r requirements.txt
```

## Testleri Çalıştırma

```bash
pytest -q
```

Beklenen mevcut sonuç:

```text
153 passed
```

## Pilot Veri Setini Çalıştırma

10 takımlı sentetik pilot veri seti için:

```bash
python3 scripts/run_pilot_10_teams.py
```

Bu script şunu üretir:

```text
output/pilot_10_teams/ao_first_elo_pilot_output.csv
```

Pilot inspection raporu için:

```bash
python3 scripts/build_pilot_inspection_report.py
```

Üretilen dosyalar:

```text
output/pilot_10_teams/pilot_inspection_report.md
output/pilot_10_teams/pilot_inspection_table.csv
```

Gerçek 2026/27 sezon öncesi 10 takımlı pilotu çalıştırmak için:

```bash
python3 scripts/run_real_pilot_10_teams.py
```

Bu çalışma gerçek UEFA ve yerel sezon verilerini kullanır. Aktif ülke gücü
`25 / 0.8 / 140` v1.1 değerlerini kullanır. Reddedilen `20 / 2.0 / 360`
adayı karşılaştırma tablosunda ayrıca gösterilir. Veri kaynakları ve dönüşümler:

```text
data/real_pilot_10_teams/SOURCES.md
```

Üretilen dosyalar:

```text
output/real_pilot_10_teams/ao_first_elo_real_pilot_output.csv
output/real_pilot_10_teams/real_pilot_review_table.csv
output/real_pilot_10_teams/real_pilot_results.md
```

## Tarihsel Backtest

`2021/22-2025/26` tarihsel UEFA backtest verisini yeniden üretmek için:

```bash
python3 scripts/build_backtest_dataset.py --refresh
```

Öncelikli parametre taramasını çalıştırmak için:

```bash
python3 scripts/run_priority_backtest.py
```

Backtest 1.176 takım-sezon ve 4.344 tamamlanmış UCL, UEL ve UECL maçı
içerir. `2025/26` seçim sırasında kullanılmayan holdout sezondur. İlk
katman sonucunda `European_History_Benchmark=28` beş sezonun tamamında
iyileşen Stage A adayıdır; domestic position ve league size verisi henüz
tam olmadığı için ana config'e otomatik olarak taşınmamıştır.

Detaylı sonuçlar:

```text
output/backtest_2021_2026/backtest_report.md
output/backtest_2021_2026/independent_parameter_checks.csv
```

### Stage B: Domestic Verili Backtest

Stage A verisini yerel lig pozisyonu ve lig takım sayısıyla zenginleştirmek için:

```bash
python3 scripts/build_backtest_stage_b.py
```

İkinci parametre backtestini çalıştırmak için:

```bash
python3 scripts/run_stage_b_backtest.py
```

Stage B aynı 1.176 takım-sezon ve 4.344 maçı kullanır. Şampiyon ve lig yolu
katılımcılarında gerekli domestic position kapsaması tamdır; üst lig dışında
kupa yoluyla gelen takımlar modelin açık unknown-position davranışında kalır.
`2025/26` yine kilitli holdout sezondur.

Stage B sonucunda ülke/lig gücü ile domestic achievement ölçeği güçlü adaylar
olarak öne çıkmıştır. Buna karşılık sezon ağırlıkları, exposure, percentile ve
kupa değerlerini değiştirmek için yeterli bağımsız kanıt oluşmamıştır. Grid
kazananı doğrudan ana modele alınmamıştır; kesin config değişikliği nested
walk-forward ve belirsizlik analizi sonrasına bırakılmıştır.

```text
output/backtest_stage_b_2021_2026/backtest_report.md
output/backtest_stage_b_2021_2026/focused_parameter_checks.csv
output/backtest_stage_b_2021_2026/independent_parameter_checks.csv
```

### Nested Walk-Forward ve Turnuva Ayrımı

Ülke gücü ve domestic achievement adaylarını yalnızca geçmiş sezonlarla seçip
bir sonraki görünmeyen sezonda test etmek için:

```bash
python3 scripts/run_nested_walk_forward.py
```

Üç expanding-window fold kullanılır: `2023/24`, `2024/25` ve `2025/26` sırayla
test sezonlarıdır. Sonuçlar ayrıca UCL, UEL ve UECL olarak ayrılır ve eşleşmiş
maç log-loss farkları bootstrap güven aralıklarıyla kontrol edilir.

Bu analizde `benchmark=20`, `gamma=2.0` ve
`domestic_league_component=360` adayı aggregate metriklerde üç görünmeyen
fold’un tamamında iyileşmiştir. Ancak sıfır-exposure Como senaryosunda
gerçek dışı zirve rating ürettiği için production adayı olarak reddedilmiştir.
Domestic achievement’ın
ülke modeline ek katkısı güvenilir olmadığı için mevcut
`domestic_achievement_component=160` ve `achievement_alpha=0.40` değerleri
korunur.

```text
output/backtest_nested_walk_forward/backtest_report.md
output/backtest_nested_walk_forward/fold_selections.csv
output/backtest_nested_walk_forward/competition_summary.csv
output/backtest_nested_walk_forward/paired_uncertainty.csv
```

### Ranking-First Kalibrasyon

Ülke gücü adaylarını maç log loss'undan önce takım sıralamasıyla değerlendirmek için:

```bash
python3 scripts/build_backtest_dataset.py \
  --start-end-year 2019 \
  --last-end-year 2026 \
  --output-root data/backtest_2018_2026

python3 scripts/build_backtest_stage_b.py \
  --stage-a-root data/backtest_2018_2026 \
  --output-root data/backtest_stage_b_2018_2026

PYTHONPATH=src python3 scripts/run_ranking_first_calibration.py \
  --data-root data/backtest_stage_b_2018_2026 \
  --output-root output/ranking_first_calibration_2018_2026
```

Bu aşama UCL, UEL ve UECL içinde Spearman sıralama korelasyonu, ikili takım
sırası doğruluğu, üst çeyrek isabeti ve percentile sıra hatasını ölçer. Adaylar
ayrıca tarihsel sıfır-exposure kontrolünü ve
`data/real_pilot_10_teams/ranking_guardrails.csv` içindeki açık takım sırası
vetolarını geçmek zorundadır. Log loss yalnızca ranking metriklerinden sonra
eşitlik bozucu olarak kullanılır.

Uzun doğrulama `2018/19-2025/26` arasındaki 8 sezonu, 1.887 takım-sezonu ve
6.340 tamamlanmış Avrupa maçını kapsar. Stage B denetiminde 8 sezonun tamamı,
428 lig tablosu ve 436 ülke-sezon katılım kontrolü geçmiştir.

245 aday altı expanding-window fold'da değerlendirilmiştir. Fold'larda farklı
adayların seçilmesi, görünmeyen sezonların yalnızca `4/6`'sında iyileşme ve UCL
ile UECL sıralama skorlarındaki düşüş nedeniyle nihai karar `KEEP_V1_1` olmuştur.
Hiçbir deneysel ülke parametresi ana modele taşınmamıştır.

```text
output/ranking_first_calibration_2018_2026/backtest_report.md
output/ranking_first_calibration_2018_2026/fold_selections.csv
output/ranking_first_calibration_2018_2026/competition_ranking_summary.csv
output/ranking_first_calibration_2018_2026/pilot_guardrail_results.csv
```

### Rating Yayılımı Kalibrasyonu

Başlangıç rating'lerinin birbirine fazla yakın olup olmadığını, takım sırasını
değiştirmeden test etmek için:

```bash
PYTHONPATH=src python3 scripts/run_rating_spread_calibration.py
```

Analiz 8 sezon, 1.887 takım-sezon ve 6.340 maç kullanır. `elo_scale` ile saha
avantajı yalnızca geçmiş sezonlarda seçilir ve sonraki görünmeyen sezonda test
edilir. Saf rating yayılımı testi, saha avantajını `70` değerinde sabit tutarak
altı fold'un tamamında `elo_scale=350` seçmiştir. Bu, standart `400` dönüşümüne
göre rating farklarının yaklaşık `%14.3` daha güçlü yorumlanması anlamına gelir.

Buna karşılık saf yayılım adayı görünmeyen sezonların yalnızca `4/6`'sında
iyileşmiş ve UEL aggregate log loss değerini `+0.000226` kötüleştirmiştir. Katı
turnuva vetosu nedeniyle karar `KEEP_CURRENT_RATING_SPREAD` olmuştur; statik
`ao_first_elo` değerleri görsel olarak açılmamıştır.

Ortak olasılık kalibrasyonunda her fold `elo_scale=300` ve saha avantajı `50`
seçmiştir. Bu çift statik rating modelinin parçası değildir; ileride geliştirilecek
maç olasılığı katmanı için tanısal aday olarak saklanır.

```text
output/rating_spread_calibration_2018_2026/spread_calibration_report.md
output/rating_spread_calibration_2018_2026/fold_selections.csv
output/rating_spread_calibration_2018_2026/competition_summary.csv
output/rating_spread_calibration_2018_2026/paired_uncertainty.csv
```

### Dinamik Elo Veri Seti ve İlk Çekirdek Kalibrasyon

Maç sonrası rating araştırması için kronolojik olay veri setini üretmek üzere:

```bash
python3 scripts/build_dynamic_backtest_dataset.py
```

Veri seti 8 sezon ve 6.340 maçı tek maç grain'inde tutar. Eleme turları
eşleşme/bacak sırasıyla, grup aşamaları altı gerçek maç günü düzeniyle, lig
aşamaları ise doğrulanmış kaynak bloklarıyla sıralanır. Penaltı sonucu maç
skorundan ayrı tutulur; penaltıya giden beraberlik maç güncellemesinde `0.5`,
tur atlayan takım ise sonraki progression araştırması için `advanced_team_id`
olarak saklanır. Takvim tarihi ve 90/120 dakika skor ayrımı kesin olmadığı
yerlerde uydurulmaz.

```text
data/dynamic_backtest_2018_2026/matches.csv
data/dynamic_backtest_2018_2026/build_audit.csv
```

Yalnızca `Scale`, saha avantajı ve temel `K` değerini kalibre etmek için:

```bash
python3 scripts/run_dynamic_core_calibration.py
```

İlk çekirdek çalışma 13 Scale, 9 saha avantajı ve 9 K değeri olmak üzere
1.053 kombinasyonu altı expanding walk-forward fold'da test eder. Her sezonda
ratingler donmuş AO First Elo v1.1 değerinden yeniden başlar; turnuva, tur,
gol farkı, progression ve season carry çarpanları bu koşuda etkisizdir.

Araştırma adayı `Scale=225`, `H=40`, `K=28` olmuştur. Aday, aynı eğitim
sezonlarında kendi Scale/H değerini seçen `K=0` statik karşılaştırmayı görünmeyen
sezonların `6/6`'sında ve UCL/UEL/UECL segmentlerinin tamamında geçmiştir.
Bu değerler production config'e henüz yazılmamıştır; sonraki katman testleri
tamamlanana kadar araştırma adayıdır.

```text
output/dynamic_core_calibration_2018_2026/calibration_report.md
output/dynamic_core_calibration_2018_2026/fold_selections.csv
output/dynamic_core_calibration_2018_2026/paired_uncertainty.csv
output/dynamic_core_calibration_2018_2026/competition_summary.csv
```

#### Gol Farkı Katmanı

Kontrollü gol farkı çarpanını çekirdekten bağımsız test etmek için:

```bash
python3 scripts/run_goal_margin_calibration.py
```

Test edilen formül:

```text
G = min(goal_cap, 1 + goal_weight * ln(goal_difference))
G = 1 for draws and one-goal results
```

Nested fold seçimlerinin tamamında `goal_weight=0.50`, `goal_cap=2.00` seçilmiş
ve katman görünmeyen sezonların `5/6`'sında çekirdeği geçmiştir. Bununla birlikte
eşleşmiş toplam Brier farkının `%95` güven aralığı sıfırı kestiği için karar
`KEEP_GOAL_MARGIN_AS_CANDIDATE` olmuştur. İyileşme yalnızca UCL segmentinde
istatistiksel olarak güvenilirdir; UEL ve UECL yönü olumlu fakat kesin değildir.

Görünmeyen-sezon duyarlılık tablosunda daha temkinli `weight=0.25` adayları
`6/6` kazanıp sıralamayı daha fazla korumuştur. Bu sonuç outer test verisi
görüldükten sonra keşfedildiğinden production parametresi olarak seçilmez;
gelecek holdout için challenger olarak saklanır. Aktif araştırma çekirdeği
`Scale=225`, `H=40`, `K=28` olarak değişmeden kalır.

```text
output/goal_margin_calibration_2018_2026/calibration_report.md
output/goal_margin_calibration_2018_2026/outer_candidate_sensitivity.csv
output/goal_margin_calibration_2018_2026/paired_uncertainty.csv
output/goal_margin_calibration_2018_2026/goal_difference_distribution.csv
```

#### Turnuva K Çarpanları

UCL'yi `1.00` referans alıp UEL ve UECL için farklı effective K değerlerini
test etmek üzere:

```bash
python3 scripts/run_competition_multiplier_calibration.py
```

Grid zorunlu `UCL >= UEL >= UECL` hiyerarşisi altında çalışır. Tam veri adayı
`UCL=1.00`, `UEL=1.00`, `UECL=0.625` olmuştur. `K=28` çekirdeğiyle bunlar
sırasıyla `28`, `28` ve `17.5` effective K üretir. Katman yalnızca `2/6`
görünmeyen fold'da çekirdekten daha iyi sonuç üretmiştir; bu nedenle karar
`KEEP_COMPETITION_MULTIPLIERS_AS_CANDIDATE` olarak korunmuştur.

Bağımsız ablation, bütün iyileşme yönünün UECL K'sını azaltmaktan geldiğini
göstermiştir. UCL ve UEL nötr `1.00` değerinde kalır; UECL-only Brier farkı
UECL'de `-0.000314` olmuştur. UECL parametresi yalnızca son üç fold'da yeterli
geçmişe dayandığı için henüz production'a alınmaz. Aktif araştırma çekirdeği
tüm turnuvalar için `K=28` kullanmaya devam eder.

Bu deney turnuva prestijini değil, maç sonucundan öğrenme hızını ölçer. Bu
nedenle `UEL=1.00` sonucu UEL'in UCL ile aynı sportif değerde olduğu anlamına
gelmez ve prestij katsayısı olarak kullanılamaz.

```text
output/competition_multiplier_calibration_2018_2026/calibration_report.md
output/competition_multiplier_calibration_2018_2026/fold_selections.csv
output/competition_multiplier_calibration_2018_2026/paired_uncertainty.csv
output/competition_multiplier_calibration_2018_2026/ablation_summary.csv
output/competition_multiplier_calibration_2018_2026/effective_k_table.csv
```

#### Maç Galibiyeti Prestij Bonusu

UCL, UEL ve UECL hiyerarşisini maç galibiyeti üzerine ayrı bir bonus olarak
eklemeyi test etmek için:

```bash
python3 scripts/run_competition_prestige_calibration.py
```

Grid katı biçimde `UCL=1.00 > UEL > UECL` koşulunu ve UEL ile UECL arasında
en az `0.10` farkı uygular. Referans aday `1.00 / 0.65 / 0.45` grid içinde
yer alır. Buna rağmen altı fold'un tamamında ek galibiyet bonusu `0`
seçilmiştir. Normal Elo güncellemesi galibiyet bilgisini zaten kullandığı için
ikinci `(S-E)` bonusu aynı kanıtı iki kez saymış ve tahmin performansını
kötüleştirmiştir. Karar `REJECT_WIN_PRESTIGE_BONUS_KEEP_CORE` olmuştur.

```text
output/competition_prestige_calibration_2018_2026/calibration_report.md
output/competition_prestige_calibration_2018_2026/fold_selections.csv
output/competition_prestige_calibration_2018_2026/full_candidate_metrics.csv
```

#### Tur Atlama Prestij Bonusu

Turnuva hiyerarşisini doğrudan galibiyete değil, eleme eşleşmesini geçme
başarısına uygulamak için:

```bash
python3 scripts/run_progression_prestige_calibration.py
```

```text
K_progression = K_core * progression_ratio
Delta = K_progression * competition_prestige * (Advanced - ExpectedToAdvance)
```

Eşleşme beklentisi ilk maçtan önce ve saha avantajı olmadan dondurulur. Normal
maç güncellemesinden sonra tur atlama güncellemesi yalnızca bir kez ve sıfır
toplamlı uygulanır. Tek maçlı ve iki maçlı eşleşmeler, penaltı kararları ve
finaller dahil 2.106 eşleşme test edilmiştir.

Nested seçim altı fold'un tamamında `progression_ratio=0` bulmuştur. Sıfır
adayın sonucu gizlememesi için her eğitim penceresindeki en iyi zorunlu pozitif
aday da ayrıca görülmemiş sezonda çalıştırılmıştır. Bu challenger `0/6` fold
kazanmış; toplam Brier farkı `+0.000268` ve `%95` güven aralığı
`+0.000157` ile `+0.000379` olmuştur. UCL, UEL ve UECL'nin üçünde de yön
zararlıdır. Karar `REJECT_PROGRESSION_PRESTIGE_KEEP_CORE` olmuştur.

Bu nedenle dinamik güç rating'inde aktif turnuva prestij katsayısı yoktur.
`1.00 / 0.65 / 0.45` geçerli bir domain sıralaması ve test referansıdır, ancak
etkin progression K sıfır olduğu için kalibre edilmiş model katsayısı değildir.

```text
output/progression_prestige_calibration_2018_2026/calibration_report.md
output/progression_prestige_calibration_2018_2026/fold_selections.csv
output/progression_prestige_calibration_2018_2026/positive_challenger_paired_uncertainty.csv
output/progression_prestige_calibration_2018_2026/progression_k_table.csv
```

#### Eleme Aşaması Çarpanları

Flat progression bonusunun erken eleme turları yüzünden başarısız olup
olmadığını bağımsız test etmek için:

```bash
python3 scripts/run_stage_progression_calibration.py
```

Eski ve yeni turnuva formatları altı ortak aşamaya normalize edilir:
`QUALIFYING`, `KNOCKOUT_PLAYOFF`, `ROUND_OF_16`, `QUARTERFINAL`, `SEMIFINAL`
ve `FINAL`. Eski formatta UCL `Round 2` son 16; UEL `Round 2` erken eleme ve
UEL `Round 3` son 16 olarak ele alınır.

Test, turnuva hiyerarşisini kalibre etmez; domain referansı olarak
`UCL=1.00`, `UEL=0.65`, `UECL=0.45` sabit tutulur. Altı önceden tanımlı,
azalmayan aşama profili ile `0-1.0` progression oranı nested walk-forward
seçiminde karşılaştırılır.

Nested seçim yine `6/6` fold'da `progression_ratio=0` bulmuştur. En iyi
zorunlu pozitif aday erken eleme ve knockout play-off bonusunu kapatıp son
16'dan itibaren küçük artışlar kullanan profiller olmuştur. Bu aday yalnızca
`3/6` fold kazanmış; toplam Brier farkı `+0.000001` ve `%95` güven aralığı
`-0.000004` ile `+0.000005` olmuştur. Sıralama guardrail'leri geçilmiş olsa da
tahmin faydası güvenilir değildir. Karar `KEEP_STAGE_PROGRESSION_AS_CANDIDATE`;
katman aktif modele alınmamıştır.

Finalden sonra verilen bir güncellemenin aynı sezonda etkileyeceği başka maç
yoktur. Sezonlar başlangıç Elo'sundan yeniden başladığı için final/trophy
çarpanı bu koşuda tanımlanabilir değildir ve `0` sabitlenmiştir. Bu değer ancak
season carry katmanıyla birlikte kalibre edilebilir.

```text
output/stage_progression_calibration_2018_2026/calibration_report.md
output/stage_progression_calibration_2018_2026/fold_selections.csv
output/stage_progression_calibration_2018_2026/positive_challenger_paired_uncertainty.csv
output/stage_progression_calibration_2018_2026/stage_multiplier_table.csv
```

#### Katı Turnuva Maç Katsayıları

Turnuva farkını ek bonus yerine doğrudan normal maç Elo güncellemesinde test
etmek için:

```bash
python3 scripts/run_strict_competition_match_calibration.py
```

```text
Delta_match = K_core * Competition Multiplier * (S - E)
```

Nötr `1/1/1` çekirdek yalnızca karşılaştırma modelidir. Test edilen bütün
adaylar `UCL=1.00 > UEL > UECL` ve en az `0.10` UEL-UECL farkı koşullarını
sağlar. `1.00 / 0.65 / 0.45` referansı ayrıca her görülmemiş sezonda bağımsız
challenger olarak çalıştırılır.

Referans aday yalnızca `2/6` fold kazanmış ve toplam Brier farkını `+0.000211`
kötüleştirmiştir. UEL segmentindeki fark `+0.000894` olmuştur. En iyi zorunlu
katı aday `1.00 / 0.95 / 0.65` değerlerini bulmuş; `3/6` fold kazanmış ve
güven aralığı sıfırı kesmiştir. Nested seçim altı fold'un beşinde nötr
`1/1/1` çekirdeği korumuştur.

Karar `KEEP_STRICT_COMPETITION_MATCH_MULTIPLIERS_AS_CANDIDATE` olmuştur ve
hiçbir turnuva maç katsayısı aktif modele yazılmamıştır. Bu sonuç turnuvaların
sportif değerini eşitlemez; `.65/.45` hiyerarşisinin normal Elo öğrenme hızına
uygulandığında fazla sert olduğunu gösterir. Prestij farkı achievement/trophy
rezervinde, season carry ile birlikte ayrıca test edilmelidir.

```text
output/strict_competition_match_calibration_2018_2026/calibration_report.md
output/strict_competition_match_calibration_2018_2026/fold_selections.csv
output/strict_competition_match_calibration_2018_2026/reference_065_045_paired_uncertainty.csv
output/strict_competition_match_calibration_2018_2026/effective_k_table.csv
```

#### Achievement Reserve ve Season Carry

Final şampiyonluğu bonusunu sonraki sezonda ölçülebilir yapmak ve sezon sonu
Power Elo bilgisinin ne kadar korunacağını test etmek için:

```bash
python3 scripts/run_achievement_carry_calibration.py
```

```text
Power_start = (1-carry) * AO_First_current + carry * Power_end_previous
Reserve_start = decay * Reserve_end_previous
AO_live = Power + Achievement Reserve
Trophy_bonus = UCL_base * competition_reference
```

Sezonluk `team_id` değerleri kalıcı olmadığı için carry öncesinde global kulüp
kimliği oluşturulur. İsim normalizasyonu, alias ve ülke koduyla 1.887 takım-sezon
satırı 506 kalıcı kulübe eşlenmiş; ülke veya anahtar çakışması bulunmamıştır.

Power carry tek başına görülmemiş sezonların `6/6`'sında çekirdeği geçmiş;
eşleşmiş Brier farkı `-0.002612` ve `%95` güven aralığı `-0.004017` ile
`-0.001286` olmuştur. Tam veri adayı `power_carry=0.85` değeridir. Carry-only
fold seçimleri `0.85-1.00` aralığında kalmış ve `0.85` üç fold ile mod değer
olmuştur. Karar `PROVISIONAL_ACCEPT_POWER_CARRY_TROPHY_REJECTED` olarak
kaydedilmiştir; bu değer henüz production config'e yazılmamıştır.

Trophy katkısı carry'den doğrudan ayrıştırılmıştır. En iyi zorunlu achievement
adayı UCL/UEL/UECL için `20/13/9` bonus ve `0.25` decay kullanmış, fakat
carry-only karşılaştırmayı yalnızca `1/6` fold geçmiştir. Incremental Brier
farkı `+0.000070` ve güven aralığı `-0.000015` ile `+0.000155` olmuştur.
Bu nedenle final/trophy reserve tahmin Elo'suna eklenmemiştir.

Gözlenen katılımcı güçleri turnuva hiyerarşisini bonus olmadan da gösterir:

```text
Ana aşama ortalaması: UCL 829.6 / UEL 747.3 / UECL 701.0
Eleme aşaması ortalaması: UCL 894.1 / UEL 805.4 / UECL 748.2
```

Harici tarihsel Elo karşılaştırması mutlak kalite kontrolünü güçlendirir, ancak
mevcut paired ablation sonuçlarının yerine geçmez. Bu amaçla kesin UEFA tarihi,
maç öncesi ClubElo snapshot'ı ve kalıcı sağlayıcı kulüp kimliği aşağıdaki ayrı
benchmarkta hazırlanmıştır.

```text
output/achievement_carry_calibration_2018_2026/calibration_report.md
output/achievement_carry_calibration_2018_2026/club_identity_audit.csv
output/achievement_carry_calibration_2018_2026/competition_strength_profile.csv
output/achievement_carry_calibration_2018_2026/trophy_incremental_paired_uncertainty.csv
output/achievement_carry_calibration_2018_2026/external_elo_benchmark_requirements.md
```

#### Kesin Tarihli Harici Elo Benchmarkı

Tekrarlanabilir veri setini üretmek ve aynı maçlarda AO ile ClubElo'yu
karşılaştırmak için:

```bash
python3 scripts/build_external_elo_benchmark.py --allow-fuzzy-team-matches
python3 scripts/run_external_elo_benchmark.py
```

Resmi UEFA maç servisi UCL `1`, UEL `14` ve UECL `2019` kimlikleriyle
kullanılmış; 2018/19-2025/26 arasındaki `6.340/6.340` maça kesin UTC başlama
zamanı ve UEFA maç/takım kimliği eklenmiştir. Kulüplerin `1.179` tanesi ad+ülke,
`708` tanesi ise tekil fikstür grafiği kanıtıyla çözülmüştür. İsim benzerliği
UEFA kimliği için tek başına kabul edilmemiştir.

Harici rating kaynağı ClubElo'dan türetilen 1 ve 15 tarihli açık snapshot
arşividir. Her maçta yalnızca maç tarihinden kesin olarak eski son snapshot
kullanılır; `31` günden eski değerler reddedilir. Bu kurallarla `492` ortak maç
oluşmuş, ilk iki sezon eğitimde bırakıldıktan sonra `363` görülmemiş maçta
expanding walk-forward karşılaştırması yapılmıştır.

```text
AO current Brier       = 0.163249
ClubElo Brier          = 0.158745
AO - ClubElo           = +0.004504
%95 paired bootstrap   = [-0.005398, +0.014634]
```

Toplam fark ClubElo yönünde olsa da güven aralığı sıfırı kestiği için genel
üstünlük kesin değildir. UCL'de ClubElo farkı `+0.014327` ve güvenilir;
UEL'de iki model yaklaşık eşit, UECL'de AO yön olarak daha iyi fakat belirsizdir.
Arşiv daha çok güçlü ve yerleşik kulüpleri kapsadığı için bu sonuç tüm eleme
takımlarına genellenmez. Ayrıca AO parametreleri aynı dönemlerle kısmen
örtüşen veride kalibre edildiğinden bu çalışma nihai untouched holdout değildir.

```text
data/external_elo_benchmark_2018_2026/matches_with_dates_and_external_elo.csv
data/external_elo_benchmark_2018_2026/exact_date_events.csv
data/external_elo_benchmark_2018_2026/build_audit.csv
output/external_elo_benchmark_2018_2026/benchmark_report.md
output/external_elo_benchmark_2018_2026/paired_predictions.csv
output/external_elo_benchmark_2018_2026/paired_uncertainty.csv
```

#### UCL Harici Fark Teshisi

UCL alt kumesindeki ClubElo farkini katmanlara ayirmak ve olasilik olcegini
bagimsiz test etmek icin:

```bash
python3 scripts/run_ucl_external_diagnostics.py
python3 scripts/run_ucl_probability_scale_calibration.py
```

Normal dinamik guncelleme static baslangic Brier'ini `0.175531` degerinden
`0.170063` degerine indirmistir. Mevcut `0.85` carry sonucu `0.171128` olup
reset modele gore `+0.001066` fark yaratmis, ancak guven araligi sifiri kesmistir.
Dolayisiyla normal K veya carry katmaninda guvenilir zarar bulunmamistir.

Canli AO rating'lerinin ClubElo ile siralama korelasyonu `0.888094`, current
modelin minimum sezon basi/sonu siralama korelasyonu `0.907974` ve maksimum
rating hareketi `136.346657` olmustur. Takim siralamasi guardrail'leri gecmistir.

UCL tahmin olasiligi icin `200-400` scale araligi nested walk-forward ile
test edilmistir. Secilen adaylar baseline `225` degerini yalnizca `2/5` fold'da
gecmis ve Brier iyilesmesi guvenilir olmamistir. Elo puanlarina veya siralamaya
dokunulmadan yapilan bu test sonucunda da `225` korunmustur.

Guncel model karari ve untouched holdout takvimi `MODEL_STATUS.md` dosyasinda
tek yerde tutulur.

```text
output/ucl_external_diagnostics_2018_2026/diagnostic_report.md
output/ucl_external_diagnostics_2018_2026/ucl_club_rank_discrepancies.csv
output/ucl_probability_scale_calibration_2018_2026/calibration_report.md
MODEL_STATUS.md
```

## PDF Dokümanları

Projede üç açıklayıcı PDF bulunur:

```text
AkilOyunu_Elo_Model_Aciklayici.pdf
AkilOyunu_Elo_Model_Kisa.pdf
output/pdf/AkilOyunu_VeriAnlamlandirma.pdf
```

- İlk PDF detaylı model spesifikasyonudur.
- İkinci PDF kısa model anlatımıdır.
- Üçüncü PDF veri ihtiyaçları ve alan sözlüğüdür.

## Kalibrasyon Notu

Model iki dış benchmark parametresi ister:

```text
Country_Strength_Benchmark
European_History_Benchmark
```

Aktif v1.1 değerleri Country Strength Benchmark `25`, gamma `0.8`, domestic
league component `140` ve European History Benchmark `20` olarak dondurulmuştur.
Bu değerler 8 sezonluk ranking-first backtestte değiştirilmek için yeterince
tutarlı kanıt oluşmadığından statik başlangıç modelinde korunur.

## Durum

Bu repo şu an v1.1 model davranışıyla:

- Modüler Python hesap motoru içerir.
- Input validation kurallarını uygular.
- Sentetik pilot veri seti ve pilot raporu içerir.
- Gerçek 10 takımlı pilot veri seti ve reddedilen ülke adayı karşılaştırması içerir.
- Ranking-first nested kalibrasyon ve hard ranking guardrail'leri içerir.
- 8 sezonluk doğrulama sonucunda aktif production config'i v1.1 olarak dondurur.
- Unit testlerle statik model, dinamik veri sözleşmesi ve çekirdek güncelleme
  davranışlarını doğrular.

Maç sonrası Elo'nun olay veri seti ve ilk çekirdek kalibrasyonu araştırma
kapsamında yer alır. Gol farkı aday olarak tutulmuş; turnuva K, maç prestiji ve
progression prestiji aktif katman olarak reddedilmiştir. Aşama duyarlı
progression ve katı turnuva maç katsayıları yalnızca araştırma adayıdır ve aktif
değildir. Power carry `0.85` geçici kabul edilen araştırma adayıdır; trophy
reserve reddedilmiştir. Kesin tarihli harici ClubElo benchmarkı tamamlanmış,
genel fark istatistiksel olarak kesin bulunmamış ve UCL segmenti sonraki
inceleme önceliği olarak belirlenmiştir. Production dinamik güncelleme motoru,
untouched gelecek holdout ve kalan cap kontrolleri tamamlanana kadar yazılmaz.
