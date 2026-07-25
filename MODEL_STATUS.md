# AO European Elo Model Durumu

Güncelleme tarihi: 2026-07-23

Aktif geliştirme sürümü: `ao-european-elo-v2.0-dev-freeze`

## Kısa Karar

AO First Elo v2 ve Dynamic Power çekirdeği geliştirme verisinde donduruldu.
Standart 1X2 çıktı ve kontrollü gol farkı katmanı aktiftir. Season carry,
progression bonus, turnuva K ve European Achievement Reserve kapalıdır. V1.1
API ve pilot değerleri regresyon karşılaştırması olarak korunur.

Bu durum tamamen kanıtlanmış production model anlamına gelmez. 2026/27 sezonu
7 Temmuz'da, model freeze tarihinden önce başladığı için tamamı untouched değildir.
Prospective değerlendirme lig aşaması ve sonrasındaki gerçek pre-match kilitlerle
yapılacaktır; qualifying ve play-off kapsam dışıdır.

## Aktif Sözleşme

```text
AO First Elo ölçeği      = 500-2000 referans bandı, final clipping yok
Aktif statik üst sınır   = 2000 (beta 0/0/0 ile yapısal saturation)
Dynamic/Live hard cap    = yok
Rating multiplier        = 3.713606654783126
Country tail beta        = 0
European tail beta       = 0
Exposure tail beta       = 0
Dynamic Elo scale        = 835.5614973262
Home advantage           = 148.5442661913
Base K                   = 103.9809863339
Season Power carry       = 0.00
1X2 draw at even         = 0.24
1X2 draw shape           = 1.00
Goal difference          = active (alpha 0.10, tau 300, GD cap 4)
Progression bonus        = inactive (base 0)
Achievement Reserve      = inactive (base 0)
Competition match K      = inactive
```

Final rating'e `min/max` clipping uygulanmaz. Ancak aktif tail beta'ları sıfır,
effective exposure tavanı `0.85` ve erişilebilir achievement maksimumu `1.08`
olduğunda AO First Elo'nun yapısal maksimumu tam `2000`dir. Dynamic/Live Elo
kırpılmadığı için 2000'i aşabilir. V2 affine dönüşümü ve Scale/H/K dönüşümü,
v1.1 sıralamasını ve aynı koşullardaki beklenen puanı yapay biçimde değiştirmez.

## Kalibrasyon Kararları

| Katman | Sonuç | Ana kanıt |
| --- | --- | --- |
| 100 tail kombinasyonu | `NO_PROMOTION` | Adjusted hedefte 0/6 fold her iki ranking metriğinde iyileşti |
| Dynamic Power | `PROMOTE` | Standart 1X2 ile 6/6; Brier farkı `-0.008929`, envelope CI tamamen negatif |
| Standart 1X2 çıktı | `PROMOTE` | Tüm turnuvalarda climatology Brier ve log-loss'tan iyi |
| Season carry | `DISABLE` | Nested 1X2 seçiminde 5/6 fold; full-data aday 0.85 olsa da kapı geçilmedi |
| Eski LOG/SQRT goal margin | `DISABLE` | Final 1X2: 3/6 Brier, forward ranking 2/5; CI sıfırı kesiyor |
| Kontrollü GD (`alpha/tau`) | `PROMOTE` | `0.10/300`: Brier 6/6, iki loss zarfı negatif, pooled ranking pozitif |
| Sıfır-toplamlı progression | `REJECT` | Nested ΔBrier `-0.000014`; pratik fayda yok, forward ranking 3/5 |
| Kontrollü GD + progression | `SHADOW` | CI iyileşme yönünde, fakat ΔBrier yalnız `-0.000260` ve ranking 1/5 |
| Competition K | `DISABLE` | Joint K+hiyerarşi seçimi unseen'da 1/6 Brier, forward ranking 1/5 |
| Achievement Reserve | `DISABLE` | Final 1X2: 2/6 Brier, forward ranking 1/5; güvenilir zarar |

Tarihsel statik dağılım 1,887 takım-sezonda yüzde 100 referans bandındadır;
medyan `1075.83`, p90 `1747.65`, min/max `651.49/1996.09` ve tam sınır
değerlerinin sayısı sıfırdır.

## Final Robustness ve Harici Benchmark Riski

Optional katmanlar nihai `carry=0`, standart 1X2 ve training-only draw mapping
tabanında yeniden test edilmiştir. Dinamik sıralama aynı sezon sonuçlarıyla
ölçülmez; sezon sonu rating yalnızca takip eden sezonun schedule-adjusted
performansına karşı değerlendirilir. 2025/26 sonu için 2026/27 kullanılmadığından
beş bağımsız forward ranking fold'u vardır.

Exact-date ClubElo paired örneğinde final UCL sonucu:

```text
Eşleşme                         171
AO Dynamic 1X2 Brier       0.601615
AO Static 1X2 Brier        0.606032
ClubElo 1X2 Brier          0.572126
AO - ClubElo               +0.029489
Dependency envelope        [-0.002932, +0.064568]
```

AO Dynamic, AO Static'e karşı UCL'de `-0.004417` iyileşmiştir. ClubElo nokta
tahmini daha iyi olsa da conservative dependency zarfı sıfırı kestiği için
final modelde “ClubElo güvenilir biçimde daha iyi” sonucu tekrarlanmamıştır.
Fark kapanmış sayılmaz: ClubElo arşivi yalnız 171 unseen UCL maçını ve ağırlıkla
yerleşik kulüpleri kapsar. UCL riski 2026/27 prospective lig-aşaması holdout'unda
ayrı izlenecektir.

## Kontrollü Gol Farkı ve Progression Deneyi

Yeni deney önceki LOG/SQRT grid'inden ayrı olarak şu kullanıcı sözleşmesini
test etmiştir:

```text
M_GD = 1 + alpha × ln(min(GD, 4)) × exp(-abs(D) / tau)
```

`alpha={0,0.05,0.10,0.15,0.20}`, `tau={150,250,300,400,500}` ve sıfır-toplamlı
UCL base bonus `{0,2,4,6,8,10,12}` altı nested outer fold'da sınanmıştır.
UEL bonusu UCL'nin `2/3`, UECL bonusu `1/3` oranıdır. AO First Elo, Scale,
H, K ve carry değiştirilmemiştir.

İlk dar gridde pre-specified `alpha=0.10/tau=300` yön olarak iyi, fakat katı
tam-sıfır ranking kapısıyla terfi edememiştir. Progression ile full modelin
ranking güvenliği `1/5` kaldığından `base_bonus=0` korunmuştur. Sonraki geniş
aramada kontrollü gol farkı progression'dan ayrı değerlendirilmiştir.

Tam rapor:

```text
output/controlled_goal_progression_backtest_2018_2026/backtest_report.md
output/goal_shadow_parameter_search_2018_2026/parameter_search_report.md
```

Genişletilmiş `13 x 9` alpha/tau taramasında fixed OOS en düşük Brier
`alpha=0.30/tau=800` ile gelmiştir; bu aday olasılık hedefinin grid sınırına
yönelmeye devam ettiğini gösterdiği için otomatik terfi gerekçesi değildir.
Ranking-first shadow adayı `alpha=0.125/tau=800` seçildi. Nested geniş tarama
ΔBrier `-0.000309`, Δlog-loss `-0.000457` ve tamamen negatif dependency
zarfları üretmiştir; seçim ilk üç fold'da pozitif alpha, son üç fold'da tekrar
alpha sıfırdır. Grid sınırı adayları otomatik terfi ettirilmemiştir.

Prospective shadow kolları `BASE`, `0.10/300`, `0.20/400` ve `0.125/800`
olarak önceden kaydedilmiştir. Temiz kanıt, 2026/27 lig aşaması katılımcılarının
tam AO First Elo dosyası hazırlandıktan ve kickoff öncesi ledger kilitleri
başladıktan sonra oluşacaktır.

Muhafazakâr ve önceden belirlenmiş `0.10/300` adayı tarihsel OOS veride
Brier'ı `6/6` fold'da iyileştirmiştir: ΔBrier `-0.000174`, Δlog-loss
`-0.000250`; dependency zarfları iki metrikte de tamamen negatiftir. Pooled
Spearman `+0.000626`, pairwise `+0.000094` iyileşmiştir. Tam sıfır toleransla
sıralama güvenliği `2/5`, ileriye dönük kaydedilen pratik eşiklerle `5/5`
fold'dur. Maksimum gol çarpanı `1.138347` olmuştur. Pratik eşikler tarihsel
sonuç görüldükten sonra belirlendiği için tek başına formal prospective kanıt
değildir. Bununla birlikte önceden belirlenmiş adayın `6/6` Brier kazanması,
negatif loss zarfları, pozitif pooled sıralama ve sınırlı `1.138347` maksimum
çarpanı birlikte değerlendirilmiş; manuel product kararıyla `0.10/300`
production'a alınmıştır. Prospective izleme devam edecektir.

## Holdout Kilidi

Makinece okunabilir sözleşme:

```text
contracts/ao_european_elo_v2.json
```

2026/27 lig aşaması ve sonrasında bu dosyadaki parametreler seçim amacıyla
değiştirilemez. `run_dynamic_live.py lock`, sonuç içermeyen fixture'dan tahmini
kickoff öncesi append-only `pre_match_log.csv` dosyasına yazar; `settle` daha sonra
sonucu işler. Retrospective `run_dynamic_elo.py` yalnızca
`replay_predictions.csv` üretir ve holdout kanıtı sayılmaz. Checkpoint dosyası
processed maçları, açık tie state'ini ve ratings checksum'unu korur.

`expected_home_score` normalize maç puanıdır (`1/0.5/0`), ev sahibi galibiyet
olasılığı değildir. Üretim logları ayrıca toplamı 1 olan gerçek H/D/A
olasılıklarını taşır. `P(H)+0.5×P(D)=expected_home_score` eşitliği korunur.
Yeni model seçimi standart üç sınıflı Brier ve log-loss kullanır; eski raporların
aynı isimli alanları yalnız legacy expected-score diagnostikleridir.

## Ana Belgeler

```text
docs/AO_EUROPEAN_ELO_V2_MODEL_CONTRACT.md
contracts/ao_european_elo_v2.json
contracts/ao_european_elo_v2_production.json
output/v2_ranking_calibration_2018_2026/calibration_report.md
output/v2_dynamic_calibration_2018_2026/calibration_report.md
output/v2_goal_margin_calibration_2018_2026/calibration_report.md
output/v2_achievement_reserve_calibration_2018_2026/calibration_report.md
output/v2_evaluation_upgrade_2018_2026/evaluation_report.md
output/v2_evaluation_upgrade_2018_2026/selected_production_model.json
output/final_robustness_2018_2026/robustness_report.md
output/final_robustness_2018_2026/robustness_manifest.json
output/controlled_goal_progression_backtest_2018_2026/backtest_report.md
output/goal_shadow_parameter_search_2018_2026/parameter_search_report.md
```
