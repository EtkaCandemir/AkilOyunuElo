# AO European Elo Model Durumu

Güncelleme tarihi: 2026-08-13

Aktif geliştirme sürümü: `ao-european-elo-v2.0-dev-freeze`

## Kısa Karar

AO First Elo v2 ve Dynamic Power çekirdeği geliştirme verisinde donduruldu.
Standart 1X2 çıktı, kontrollü gol farkı ve sabit kazanan-only progression
katmanları aktiftir. Season carry, sıfır-toplamlı progression, turnuva K ve
European Achievement Reserve kapalıdır. V1.1 API ve pilot değerleri regresyon
karşılaştırması olarak korunur.

Pre-match kullanıcı tahmini ayrıca `PROMOTE_WITH_MONITORING` statüsündedir:
mevcut Structural ML olasılığı ile `rho=0` AO Domestic Poisson olasılığı
log-probability uzayında `%50/%50` birleştirilir. Bu katman ratinge geri
beslenmez; artifact, feature veya state sorunu olduğunda Current AO 1X2'ye
döner ve 2026/27 boyunca her tahmin karşılaştırmalı kaydedilir.

Beş sezonluk varyans kontrollü Domestic Surprise manuel ürün kararıyla AO First
Elo production pipeline'ına alınmıştır. Aktif değerler `theta=0.40`,
`gamma=0.50`, Domestic Prior tavanı `+/-30` ve minimum tarihçe `5` tam sezondur.
Eksik tarihçe düzeltme üretmez.

Manuel ürün kararıyla `goal_alpha=0.15` ve bounded xG
`ratio=0.30/scale=1.25` birleşimi production runtime'a alınmıştır. İki tarafın
doğrulanmış xG'si yoksa maç `alpha=0.15` gol farkı güncellemesine geri döner.
Beraberlik ve penaltı shootout kararı xG bonusu üretmez.

Bu durum tamamen kanıtlanmış production model anlamına gelmez. 2026/27 sezonu
7 Temmuz'da, model freeze tarihinden önce başladığı için tamamı untouched değildir.
Prospective değerlendirme lig aşaması ve sonrasındaki gerçek pre-match kilitlerle
yapılacaktır; qualifying ve play-off kapsam dışıdır.

## Aktif Sözleşme

```text
AO First Elo ölçeği      = 500-2000 referans bandı, final clipping yok
Aktif statik üst sınır   = 2030 (Domestic Surprise +30 dahil)
Dynamic/Live hard cap    = yok
Rating multiplier        = 3.713606654783126
Country tail beta        = 0
European tail beta       = 0
Exposure tail beta       = 0
Dynamic Elo scale        = 835.5614973262
Home advantage           = 148.5442661913
Base K                   = 103.9809863339
Dynamic K                = inactive; fixed K korunuyor
Domestic Surprise        = active (theta 0.40, gamma 0.50, cap +/-30)
Season Power carry       = 0.00
1X2 draw at even         = 0.24
1X2 draw shape           = 1.00
Single-match tie draw    = 0.12 (format metadata; rating state unchanged)
Goal difference          = active (alpha 0.15, tau 300, GD cap 4)
xG performance           = active (ratio 0.30, scale 1.25, analytic minimum gain ratio 0.70; not a runtime clamp)
Progression bonus        = active (R16 ve sonrası 12/8/4; cap 48/32/16)
Achievement Reserve      = inactive (base 0)
Competition match K      = inactive
Production prediction    = active with monitoring (50% Current ML + 50% AO Domestic Poisson)
Poisson rho              = 0
Prediction fallback      = Current AO 1X2
Prediction -> Elo        = disabled
```

## Nihai Model Adayı

```text
Candidate status         = FINAL_MODEL_CANDIDATE
Candidate version        = ao-european-elo-v2.0-final-candidate-2026-08-13
Domestic Surprise        = active (theta 0.40, gamma 0.50, cap +/-30)
Goal difference          = active (alpha 0.15, tau 300, GD cap 4)
xG performance           = active (ratio 0.30, scale 1.25)
Progression bonus        = active (R16 ve sonrası 12/8/4; cap 48/32/16)
Winner minimum ratio     = 0.70 x classic result gain, analytically implied by xG ratio; runtime floor is non-binding
Missing xG fallback      = alpha 0.15 goal-margin-only update
Draw / penalty shootout  = xG adjustment 0
Power update             = zero-sum
Match movement hard cap  = yok
Production activation    = true
Prediction activation    = PROMOTE_WITH_MONITORING
```

Makinece doğrulanan aday sözleşmesi:

```text
contracts/ao_european_elo_v2_final_candidate.json
```

Final rating'e `min/max` clipping uygulanmaz. Aktif tail beta'ları sıfırken
Domestic Surprise öncesi AO First Elo yapısal maksimumu `2000`, `+30` sürpriz
düzeltmesiyle yeni maksimum `2030`dur. Dynamic/Live Elo kırpılmadığı için bu
değeri aşabilir. Sürpriz kapalıyken V2 affine dönüşümü ve Scale/H/K dönüşümü,
v1.1 sıralamasını ve aynı koşullardaki beklenen puanı yapay biçimde değiştirmez.

## Kalibrasyon Kararları

| Katman | Sonuç | Ana kanıt |
| --- | --- | --- |
| 100 tail kombinasyonu | `NO_PROMOTION` | Adjusted hedefte 0/6 fold her iki ranking metriğinde iyileşti |
| Domestic Surprise | `PROMOTE_MANUAL` | Cap-30 gamma testinde gamma 0.50 sıralamayı 6/6 korudu; 2025/26 replay Brier `-0.000622`, log-loss `-0.000910`, pooled ranking `+0.001303` |
| Dynamic Power | `PROMOTE` | Standart 1X2 ile 6/6; Brier farkı `-0.008929`, envelope CI tamamen negatif |
| Standart 1X2 çıktı | `PROMOTE` | Tüm turnuvalarda climatology Brier ve log-loss'tan iyi |
| Tek maç format düzeltmesi | `ACTIVATE_STRUCTURAL` | Full fit `0.1129`; sabit `0.12` ile pooled Brier `-0.000658`, log-loss `-0.001617`; 200/248 maç 2020/21 olduğundan kanıt sınırlılığı ayrıca işaretli |
| Alt-1 beraberlik shape | `KEEP_SHADOW` | Full fit `0.24/0.84`; walk-forward Brier `4/6`, log-loss `3/6`, pooled farklar `+0.000036/+0.000243` |
| Season carry | `DISABLE` | Nested 1X2 seçiminde 5/6 fold; full-data aday 0.85 olsa da kapı geçilmedi |
| Eski LOG/SQRT goal margin | `DISABLE` | Final 1X2: 3/6 Brier, forward ranking 2/5; CI sıfırı kesiyor |
| Kontrollü GD (`alpha/tau`) | `PROMOTE` | `0.10/300`: Brier 6/6, iki loss zarfı negatif, pooled ranking pozitif |
| Sabit kazanan-only progression | `PROMOTE_MANUAL` | R16 ve sonrası `12/8/4`; KPO rota asimetrisi nedeniyle dışarıda, cap `48/32/16` |
| Kontrollü xG ile GD alpha taraması | `NO_AUTOMATIC_PROMOTION` | `0.10-0.25` gridinde nested unseen Brier `+0.000136`, log-loss `+0.000202`; yüksek alpha UCL/UEL'i az iyileştirirken UECL'i kötüleştirdi |
| `alpha=0.15` + xG tam sezon replay | `FINAL_MODEL_CANDIDATE` | 961 maçta `0.10+xG`ye karşı Brier `-0.000002`, log-loss `+0.000010`; ranking `+0.001941`, pairwise `+0.000720`, final medyan mutlak fark `1.03` Elo |
| xG harmanı | `SHADOW_ONLY` | 606 FotMob maçında loss iyileşti; ranking-first guardrail hafif geriledi ve tek sezon kanıtı terfi için yetersiz |
| xG unsupported margin | `SHADOW_ONLY` | Temporal validation'da Brier `+0.000427`, log-loss `+0.000280`; ranking ve UCL segmenti geriledi |
| xG goal-bonus guard | `SHADOW_ONLY` | Yalnız GD bonusunu düzenledi; Brier `+0.000027`, pairwise `+0.000103`, güvenilir zarar yok fakat terfi kapısı geçilmedi |
| xG çift yönlü performans bonusu | `SHADOW_CANDIDATE` | Beş nested foldun tamamında seçildi; pooled Brier `-0.005023`, log-loss `-0.007137` ve sıralama olumlu, fakat UECL geriledi ve cluster CI sıfırı kesti |
| Kontrollü xG düzeltmesi | `PROMOTE_MANUAL` | `%30` etki tavanlı `ratio=0.30/scale=1.25`; sabit unseen Brier `-0.002542`, log-loss `-0.003828`, üç turnuva olumlu; production'da aktif |
| ML + Domestic Poisson 1X2 | `PROMOTE_WITH_MONITORING` | 4.884 unseen maçta Brier `0.568093`, log-loss `0.959242`; AO'ya fark `-0.003999/-0.005129`; `%50/%50`, `rho=0`, AO fallback, rating feedback kapalı |
| Takım belirsizliğine göre Dynamic K | `KEEP_FIXED_K` | Nested ΔBrier `+0.000085`, Δlog-loss `+0.000112`; forward ranking güvenli 1/5 |
| Format-duyarlı `P_advance` | `SHADOW_ONLY` | Tie Brier `-0.003531`, log-loss `-0.009814`; bazı turnuva/format segmentleri geriledi |
| Sıfır-toplamlı progression | `REJECT` | Nested ΔBrier `-0.000014`; pratik fayda yok, forward ranking 3/5 |
| Kontrollü GD + progression | `SHADOW` | CI iyileşme yönünde, fakat ΔBrier yalnız `-0.000260` ve ranking 1/5 |
| Achievement Reserve, kalibre `P_advance` | `DISABLE` | Güncel comparator'a karşı ΔBrier `+0.004836`; güvenilir zarar ve ranking 0/5 |
| Competition K | `DISABLE` | Joint K+hiyerarşi seçimi unseen'da 1/6 Brier, forward ranking 1/5 |
| Rövanş toplam skor bağlamı | `NO_PROMOTION` | Yedi adayda ranking-first seçim her fold baseline `0` değerini korudu |
| Dinamik saha avantajı | `NO_PROMOTION` | 108 profilde ranking-first seçim her fold global saha avantajını korudu |
| Takım bazlı home/away residual context | `KEEP_SHADOW_CANDIDATE` | Prediction-only nested testte loss 5/6 foldda iyileşti; pooled Brier `-0.000885`, log-loss `-0.001265`, fakat dependency CI sıfırı kesti |
| Bağlamsal beraberlik | `NO_PROMOTION` | ΔBrier `+0.000850`, Δlog-loss `+0.001640`; yalnız 1/6 fold kazandı |
| Domestic-Prior sezon regresyonu | `NO_PROMOTION_LATEST_CONTEXT_REPLAY` | Ranking vetosu CI-temelli düzeltildi ve geçiyor; güncel match-context replay'inde ΔBrier `-0.003339`, Δlog-loss `-0.004885`, fakat Brier 3/6 fold ve conservative üst sınır `+0.000110` |
| Achievement Reserve | `DISABLE` | Final 1X2: 2/6 Brier, forward ranking 1/5; güvenilir zarar |

Domestic Surprise öncesi tarihsel statik dağılım 1,887 takım-sezonda yüzde 100 referans bandındadır;
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

## Dynamic K Backtesti

Takım belirsizliğine göre K değişimi, aktif `0.10/300` kontrollü gol farkı
tabanına karşı ayrı test edilmiştir:

```text
U_exposure   = 1 - European Exposure
U_matches    = exp(-Prior Matches / Match Evidence Scale)
U_inactivity = min(Days Since Last Match / Inactivity Days, 1)
U_team       = (U_exposure + U_matches + U_inactivity) / 3
K_team       = min(K_base x K Cap, K_base x (1 + lambda x U_team))
K_match      = arithmetic veya geometric aggregate(K_home, K_away)
```

`lambda={0,0.15,0.30,0.50}`, K cap `{1.25,1.50,1.75}`, iki aggregation,
inactivity `{180,270,365}` ve match evidence `{6,10,15}` ile 163 tekil aday,
6 nested outer fold ve 6.340 exact-date maçta test edilmiştir. Unseen
karşılaştırma 4.884 maçtır.

Nested Dynamic K sonucu baseline'a göre Brier'da `+0.00008489`, log-loss'ta
`+0.00011225` ile kötüleşmiştir. Yalnızca `2/6` fold loss kazanmış, iki
forward ranking metriğinde gerilemeyen fold sayısı `1/5` kalmıştır. UCL ve
UEL nokta tahminleri küçük iyileşirken UECL Brier `+0.00046005` kötüleşmiştir.
Düşük exposure bandında ortalama K `122.64`e çıkmış ve Brier farkı
`+0.001046` olmuştur. Fixed yüzeyde 70 aday iki loss metriğini çok küçük
iyileştirse de 162 dinamik adayın hiçbiri pooled ranking guardrail'ini
geçmemiştir.

Bu nedenle production `K=103.9809863339` sabit kalır; Dynamic K aktif veya
shadow moda alınmaz. AO First Elo ve diğer production parametreleri
değişmemiştir.

Tam rapor:

```text
output/dynamic_k_backtest_2018_2026/backtest_report.md
```

## Gol Farkı ve xG Ablation Backtesti

xG, AO First Elo'yu değiştirmeden yalnız maç sonrası Power güncellemesinde
test edilmiştir. `BASE`, `GD`, `XG` ve `GD_XG` kolları aynı başlangıç Elo,
`K=103.9809863339`, Scale ve saha avantajıyla karşılaştırılmıştır. Eksik xG
sıfırla doldurulmamış; yalnız eşleşen 90 dakika FT maçları kullanılmıştır.

Açık kaynakta iki taraf xG bulunan 787 maçın 426'sı AO kimliğiyle eşleşmiş,
417'si katı uygunluk sözleşmesini geçmiştir. Parametreler 2025/26 öncesinde
seçilmiş, 180 maçlık 2025/26 örneklemi temporal holdout olarak tutulmuştur.
Seçilen birleşik shadow adayı `rho=0.05`, `c_xG=1.00` olmuştur.

Holdout'ta `GD_XG`, aktif `GD` modeline göre Brier'da `-0.000680`,
log-loss'ta `-0.000950` üretmiş ve clustered envelope iki metrikte de negatif
kalmıştır. UCL, UEL ve UECL nokta tahminleri iyileşmiştir. Buna karşılık pooled
Spearman `-0.000540`, pairwise `-0.000257` gerilemiştir.

Kaynak xG, per-shot Opta/StatsBomb metriği değil; şut bölgelerinden türetilmiş
kaba API-Football tahminidir ve sezon-turnuva kapsamı kesintilidir. Bu nedenle
xG production'a alınmamış, yalnız shadow araştırma adayı olarak bırakılmıştır.
Aktif model kontrollü gol farkı ve sabit K kullanmaya devam eder.

Daha sonra 2025/26'nın 606 doğrulanmış FotMob shot-based xG maçıyla, önceki
parametreler yeniden seçilmeden ikinci bir doğrulama yapılmıştır. Convex ve
yön-korumalı `rho=0.05`, `c_xG=1.00` kollarında 961 maçlık Brier farkları
sırasıyla `-0.000346/-0.000353`, log-loss farkları
`-0.000507/-0.000516` olmuş; dört bağımlılık-duyarlı loss güven aralığının
tamamı sıfırın altında kalmıştır. Buna karşılık aynı sezon Spearman
`-0.001929`, pairwise `-0.000978` gerilemiştir. Additive `rho=0.50`,
`c_xG=0.75` daha büyük nokta iyileşmesi üretse de CI sıfırı kesmiş ve Spearman
`-0.024098` düşmüştür. Ranking-first kararına göre üçü de
`MIXED_OR_INCONCLUSIVE`; production değişmemiştir.

Kullanıcı hedefiyle daha uyumlu olan `UNSUPPORTED_MARGIN` formülü de ayrıca
uygulanmıştır. Bu formül desteklenen galibiyetlere dokunmaz; yalnız gerçek gol
farkının kazananın xG üstünlüğünü ve toleransı aşan bölümünde aktif GD update'ini
azaltır. `1 Ocak 2026` öncesi 399 xG maçı development, sonraki 207 tam kapsamlı
xG maçı validation olarak ayrılmıştır. Development'ın seçtiği
`tolerance=1.00`, `lambda=0.20`, `minimum_multiplier=0.70` adayı validation'da
Brier `+0.000427`, log-loss `+0.000280`, Spearman `-0.003238`, pairwise
`-0.001698` üretmiştir. UECL ve UEL yönü olumlu, UCL yönü olumsuzdur. 36 adayın
hiçbiri development ranking guardrail'ini geçmediği için sonuç
`MIXED_OR_INCONCLUSIVE_SHADOW` ve production kapalıdır.

İstenen güvenlik mimarisi için xG'nin tüm update'i değil yalnız
`K*(S-E)*(M_GD-1)` gol farkı bonusunu düzenlediği ayrı bir test yapılmıştır.
Development seçimi `tolerance=1.00`, `lambda=0.10`, minimum bonus çarpanı
`0.75` olmuştur. Validation'da 56 çok farklı maçın bonusu ortalama `%6.7`
azalmış; accuracy aynı, pairwise `+0.000103`, ECE olumlu, Brier `+0.000027`,
log-loss `+0.000029` ve Spearman `-0.000290` olmuştur. Güven aralığı güvenilir
zarar göstermese de tüm promotion kapıları geçmediği için katman shadow kalır.
Bu goal-bonus guard kolunda temel galibiyet/mağlubiyet Elo'su xG tarafından
azaltılmaz.

Son kullanıcı sözleşmesine uygun daha kapsamlı çift yönlü performans bonusunda
xG, temel maç sonucunun yerine geçmez. `Delta_base + Delta_GD` üzerine
`beta * abs(Delta_base) * tanh((xG_home-xG_away)/xG_scale)` eklenir; ters xG
kazancı azaltabilir, fakat yön-koruma tabanı kazanan takımın Elo değişimini her
zaman pozitif tutar. 201 aday beş expanding walk-forward foldda sınanmıştır.
xG adayı `5/5` foldda seçilmiş ve 387 unseen maçta Brier `-0.005023`, log-loss
`-0.007137`, pooled sıralama `+0.052620`, pairwise `+0.018818` üretmiştir.
UCL/UEL olumlu, UECL Brier `+0.003886` ve log-loss `+0.003691` olumsuzdur;
cluster CI sıfırı keser. Bu nedenle production değişmez. 2026/27 prospective
shadow adayı `beta=1.50`, `xG_scale=3.00`, minimum kazanan kazancı `%5` olarak
kaydedilmiştir. Kontrollü pilotta bu adayın temel kazancı `%5` tabanına kadar
indirebildiği görüldüğünden artık tercih edilen shadow sözleşmesi değildir.

Yerine xG etkisini klasik maç sonucu Elo'sunun en fazla `%15-%30`u ile
sınırlayan kontrollü aile test edilmiştir. Yirmi sekiz aday ve production
kontrolü beş expanding fold üzerinde Brier-first, ranking guardrail'li seçimle
karşılaştırılmıştır. Gelecek sezon için sabit shadow seçimi
`max_xg_ratio=0.30`, `xG_scale=1.25`tir; teorik kazanan aralığı klasik Elo'nun
`%70-%130`udur ve gol farkı bonusu ayrı kalır. Sabit aday 387 unseen maçta
Brier'ı `0.591137 → 0.588595`, log-loss'u `0.990346 → 0.986518`, doğruluğu
`%52.71 → %53.75` yönünde iyileştirmiştir. UCL, UEL ve UECL'nin tamamında
Brier/log-loss yönü olumludur; sıralama `+0.020024`, pairwise `+0.008453`tir.
Cluster CI Brier için `[-0.005760, +0.000792]` olduğundan güvenilir iyileşme
henüz kanıtlanmamıştır. Aday `PROSPECTIVE_SHADOW` statüsündedir; production
değişmemiştir.

Seçilen katsayı kontrollü pilotta ve 961 maçlık 2025/26 tam sezon replay'inde
ayrıca gösterilmiştir. Dengeli 1-0 galibiyet klasik `51.99` Elo üretirken güçlü
xG desteğinde `66.37`, ters xG'de `38.05`, uç ters xG'de `36.42` üretir. Tam
sezonda 606 xG kapsamlı maç kullanılmış; tüm maçlarda Brier
`0.589573 → 0.588284`, log-loss `0.989154 → 0.987275`, accuracy
`%53.07 → %53.49` olmuştur. Ortalama mutlak xG düzeltmesi `6.19`, uç kazanan
düzeltmeleri `+24.86/-23.10` Elo'dur. Sezon sonu production farkının medyanı
`0.44`, yüzde 90 değeri `31.10`, maksimumu `79.02` Elo'dur. UCL/UEL yönü
olumlu, tüm sezon UECL Brier farkı `+0.000129` ile çok küçük zarar yönündedir.
Bu çalışma retrospective counterfactual olduğundan production statüsünü
değiştirmez.

Tam veri ve rapor:

```text
data/xg_backtest_2018_2026/README.md
output/xg_goal_ablation_backtest_2018_2026/backtest_report.md
output/xg_goal_ablation_backtest_2018_2026/selected_xg_model.json
output/fotmob_xg_backtest_2025_26/backtest_report.md
output/fotmob_xg_backtest_2025_26/selected_shadow_model.json
output/unsupported_margin_backtest_2025_26/backtest_report.md
output/unsupported_margin_backtest_2025_26/selected_unsupported_margin_model.json
output/xg_goal_bonus_guard_backtest_2025_26/backtest_report.md
output/xg_goal_bonus_guard_backtest_2025_26/selected_goal_bonus_guard_model.json
output/xg_performance_bonus_walk_forward_2025_26/backtest_report.md
output/xg_performance_bonus_walk_forward_2025_26/selected_xg_performance_bonus_model.json
output/bounded_xg_adjustment_walk_forward_2025_26/backtest_report.md
output/bounded_xg_adjustment_walk_forward_2025_26/selected_bounded_xg_model.json
output/bounded_xg_adjustment_pilot/pilot_report.md
output/bounded_xg_full_season_replay_2025_26/replay_report.md
output/bounded_xg_full_season_replay_2025_26/final_ratings_comparison.csv
output/goal_alpha_with_bounded_xg_backtest_2025_26/backtest_report.md
output/goal_alpha_with_bounded_xg_backtest_2025_26/selected_goal_alpha_model.json
```

Kontrollü xG adayı sabitken gol farkı görünürlüğü ayrıca izole edilmiştir.
`goal_alpha=0.10/0.125/0.15/0.175/0.20/0.225/0.25` gridinde xG
`ratio=0.30/scale=1.25`, `tau=300` ve `GD cap=4` sabit kalmıştır. Bu teste maç
başına `80 Elo` sınırı veya başka bir hareket tavanı eklenmemiştir. Nested
seçim 387 unseen maçta baseline'a karşı Brier `+0.000136`, log-loss
`+0.000202`, ranking `-0.002809` üretmiştir; yalnız iki fold non-baseline alpha
seçmiştir. Bu deneyin otomatik kararı o tarihte `alpha=0.10`u korumaktı.

Retrospektif sabit aday tablosunda `alpha=0.25` pooled Brier'ı `-0.000377`
iyileştirmiştir; ancak UCL ve UEL'deki küçük kazanca karşı UECL Brier
`+0.000585`, log-loss `+0.000804` kötüleşmiş ve cluster CI sıfırı kesmiştir.
Bu seçim unseen sonuçlara baktığı için production kanıtı değildir. Yalnız geçmiş
tam-veri seçiminin önerdiği `alpha=0.15` önce prospective shadow adayı olarak
kaydedilmiş, daha sonra aşağıdaki manuel ürün kararıyla production'a alınmıştır.

Kullanıcı tarafından dengeli tasarım adayı olarak seçilen `alpha=0.15`, bounded
xG `ratio=0.30/scale=1.25` ile 961 maçlık ayrı tam sezon replay'de
çalıştırılmıştır. `0.10+xG`ye göre tüm maç Brier farkı `-0.000002`, log-loss
farkı `+0.000010` ile pratik olarak sıfırdır. Ranking `+0.001941`, pairwise
`+0.000720` iyileşmiştir. Maç başına ortalama mutlak alpha etkisi `0.47`, sezon
sonu medyan mutlak takım farkı `1.03` ve maksimum fark `7.81` Elo'dur. UCL ve
UEL loss yönü olumlu, UECL küçük zarar yönündedir; cluster CI sıfırı keser.
Replay adayın kontrollü ve sıralama bakımından olumlu olduğunu göstermiştir.
Tek başına otomatik terfi kanıtı sayılmamış, son aktivasyon manuel ürün kararıyla
yapılmıştır; 2026/27 prospective izleme zorunluluğu sürer.

```text
output/goal_alpha_015_xg_full_season_replay_2025_26/replay_report.md
output/goal_alpha_015_xg_full_season_replay_2025_26/replay_manifest.json
output/goal_alpha_015_xg_full_season_replay_2025_26/final_ratings_comparison.csv
```

## Tur Geçme Olasılığı ve Achievement Reserve

Achievement Reserve'in önceki testlerinde eşleşme öncesi tur geçme olasılığı
tek maçlık nötr Elo olasılığıyla temsil ediliyordu. Bu teknik varsayımı
kontrol etmek için 1.606 tamamlanmış eşleşmede, yalnız geçmiş sezonları
kullanan expanding walk-forward kalibrasyon uygulanmıştır:

```text
logit(P_advance) =
    slope x logit(P_raw)
    + single_match_home_bias
    + two_leg_first_home_bias
```

Format-duyarlı aday, identity proxy'ye göre tie Brier'ı `0.207211`den
`0.203680`e, log-loss'u `0.602071`den `0.592258`e indirmiştir. Clustered
zarflar toplam örneklemde iyileşme yönündedir. Buna karşılık bazı
turnuva/format segmentleri gerilediği için bu kalibrasyon production
`P_advance` olarak terfi ettirilmemiş, yalnız reserve hipotezini daha güçlü
bir olasılıkla yeniden sınamak için shadow girdi olarak kullanılmıştır.

Güncel reserve testi sabit `K=103.9809863339`, kontrollü gol farkı
`alpha=0.10`, `tau=300`, `cap=4`, carry `0` ve standart 1X2 comparator
üzerinden 961 adayla yürütülmüştür. Nested seçimde non-zero reserve her
foldda seçilmesine rağmen yalnız `1/6` fold Brier kazanmış, ileri sıralama
guardrail'i `0/5` kalmıştır:

```text
Baseline Brier       = 0.575188
Reserve Brier        = 0.580024
Delta Brier          = +0.004836
Delta log-loss       = +0.007107
Conservative Brier CI = [+0.002589, +0.007525]
```

UCL, UEL ve UECL'nin üçünde de nokta tahmini kötüleşmiştir. Full-data gridde
çok küçük iyileşme gösteren bazı non-zero adaylar bulunsa da nested
walk-forward sonuçları bunların genellenmediğini göstermiştir. Sonuç:
Achievement Reserve kapalıdır. Tur başarısının maçlarla Power Elo'ya zaten
taşındığı, ayrı reserve'in aynı kanıtı yeniden saydığı değerlendirilmiştir.

Tam raporlar:

```text
output/progression_probability_2018_2026/calibration_report.md
output/progression_probability_2018_2026/selected_progression_probability.json
output/progression_reserve_final_backtest_2018_2026/backtest_report.md
output/progression_reserve_final_backtest_2018_2026/decision.json
```

## Maç Bağlamı ve Sezon Başlangıcı Backtesti

Dört ek özellik mevcut 6.340 exact-UTC maçta, sabit K ve aktif kontrollü gol
farkı baseline'ına karşı ayrı nested walk-forward testlerden geçirilmiştir.
Toplam skor ve saha avantajı beklenen skoru, beraberlik profili yalnız 1X2
dağılımını, Domestic-Prior regresyonu yalnız sezon başlangıcını etkiler.

Rövanş toplam skor katmanı için önceki maçın saha skoru gelecek bilgi
kullanılmadan ev sahibi perspektifine çevrilmiştir:

```text
D_context = D_rating - c_aggregate x clip(aggregate_home_lead, -3, 3)
```

`c_aggregate={0,25,50,75,100,150,200}` gridinde bütün foldlar baseline
`0` değerini seçmiştir. UCL/UEL/UECL, eleme ve rövanş çarpanlarından oluşan
108 saha avantajı profilinde de bütün foldlar global baseline'a dönmüştür.
Bu iki katman production'a alınmamıştır.

Bağlamsal beraberlik gridinde full-data aday `draw_at_even=0.26` ve rövanş
offset'i `-0.02` görünse de unseen sonuç Brier'da `+0.000850`, log-loss'ta
`+0.001640` kötüleşmiş ve yalnız `1/6` fold kazanmıştır. Mevcut
`draw_at_even=0.24`, `draw_shape=1.00` korunur.

Domestic-Prior regresyon adayı şu formülle test edilmiştir:

```text
Season Start = Current Domestic Prior
             + persistence x (Previous Power Elo - Current Domestic Prior)
```

Tarihsel değerlendirme artefaktında nested aday Brier'da `-0.004535`,
log-loss'ta `-0.006590` iyileşmiş; conservative Brier aralığı
`[-0.008741,-0.000599]` olmuştur. Buna karşın tek unseen ranking foldundaki
Spearman `-0.001292`, pairwise `-0.001314` farkı belirsizlik ölçülmeden mutlak
veto olarak uygulanmıştır. Bu karar kuralı düzeltilmiştir: ranking farkları
artık target-season cluster bootstrap ve küçük örneklem t-aralığıyla ölçülür;
veto yalnız conservative yüzde 95 aralığı tamamen zarar yönündeyse çalışır.

Başarısız ilk ranking foldu ayrıca takım seviyesinde incelenmiştir. Kaynak
rating sezonu 2020/21, forward hedef 2021/22 ve 2021/22 aynı zamanda UECL'nin
ilk sezonudur. Aday UCL'de Spearman'ı `+0.051153`, pairwise metriğini
`+0.017148` iyileştirirken yeni UECL segmentinde sırasıyla `-0.017878` ve
`-0.004660` gerilemiştir. The New Saints, Sutjeska Niksic veya Dinamo Minsk
gözlemlerinden herhangi biri tek başına çıkarıldığında pooled iki sıralama
farkı da yeniden sıfırın üstüne çıkmaktadır.

Bu nedenle gap geniş ve model-geneli bir çöküş değil; ilk UECL sezonunda az
sayıda oynak kulübe duyarlı, küçük bir sıralama kaybıdır ve artık otomatik
veto değildir. Güncel kodla yeniden çalıştırılan tarihsel match-context
baseline'ında (`GD=0.10`, xG yok) ranking kapısı geçmiştir:
ortalama Spearman farkı `+0.002560`, conservative
aralık `[-0.012265,+0.017384]`; güvenilir zarar yoktur. Ancak güncel loss
sonucu tarihsel artefakttan daha zayıftır: Brier `-0.003339`, log-loss
`-0.004885`, Brier fold kazanımı `3/6` ve conservative Brier üst sınırı
`+0.000110` olmuştur. Dolayısıyla güncel karar yine `NO_PROMOTION`dur, fakat
gerekçe artık ranking gürültüsü değil loss kapılarının tamamlanmamasıdır.

Tam rapor:

```text
output/match_context_backtest_2018_2026/backtest_report.md
output/match_context_backtest_2018_2026/decision_manifest.json
output/domestic_regression_diagnostics_2018_2026/diagnostic_report.md
output/domestic_regression_diagnostics_2018_2026/diagnostic_manifest.json
```

## Reddedilen Özelliklerde Heterojenlik Denetimi

Global ret kararlarının birkaç kulüp outlier'ı yüzünden oluşup oluşmadığı,
aynı 4.884 unseen maç üzerinde leave-one-team, leave-one-season,
leave-one-fold ve leave-one-competition analizleriyle kontrol edilmiştir.

Sonuçlar:

- Achievement Reserve, Competition K ve bağlamsal beraberlik `BROAD_SYSTEMATIC_HARM` sınıfındadır. Üç turnuvanın tamamında ve altı foldun beşinde zarar üretirler; hiçbir tek takım, sezon veya turnuva çıkarıldığında loss yönü tersine dönmez.
- Aggregate State ve Dynamic Home, outlier nedeniyle reddedilmemiştir. Ranking-first eğitim seçimi altı foldun tamamında non-zero aday yerine baseline'ı seçmiştir.
- Dynamic K'nın pooled zararı küçüktür ve UECL çıkarıldığında iki loss metriği de iyileşme yönüne döner. Bu bireysel takım outlier'ı değil, turnuva-segment heterojenliğidir. UCL/UEL faydası post-hoc ve çok küçük olduğu için production'a alınmaz.
- Eski sıfır-toplamlı progression adayının pooled faydası yalnız `-0.000014` Brier'dır; 2025/26 veya UECL çıkarıldığında yön tersine döner. Bu bulgu, ayrı sabit kazanan-only `12/8/4` katmanının kanıtı olarak kullanılmaz.
- Domestic regression loss tarafında UCL/UEL/UECL'nin üçünde de iyidir; reddi loss outlier'ından değil, daha önce belgelenen ilk UECL sezonundaki forward-ranking guardrail'inden gelir.

Bu denetimde global zarar üreten hiçbir katmanın kararı tek bir kulübü
çıkarmakla değişmemiştir. Segment sonucu üzerinden geriye dönük aktivasyon
yapılmaz; segment kuralı ancak önceden tanımlanıp yeni nested walk-forward
testini geçerse değerlendirilebilir.

```text
output/rejected_feature_heterogeneity_2018_2026/audit_report.md
output/rejected_feature_heterogeneity_2018_2026/audit_manifest.json
```

## Poisson/Dixon-Coles Skor Katmanı

Mevcut production AO First Elo ve AO Live Elo akışına dokunmayan ayrı skor
tahmin katmanı, 2018/19-2025/26 verisindeki 6.340 maç ve altı expanding outer
fold ile test edilmiştir. Out-of-sample kapsam 4.884 maçtır; 2026/27 seçimden
dışlanmıştır.

```text
Full-data mu             0.30181296
Full-data beta           0.70649057
Full-data rho            0.00
Score NLL fold wins      6/6
1X2 Brier fold wins      2/6
1X2 log-loss fold wins   2/6
Pooled Brier delta      +0.001388
Pooled log-loss delta   +0.002290
```

Exact-score NLL, ev ve deplasman gol ortalamalarını ayrı öğrenen güçlü
intercept-only Poisson kontrolünü güvenilir biçimde geçmiştir. Ancak türetilen
1X2 çıktısı mevcut AO 1X2 katmanını geçmemiş; UCL gol beklentisi yanlılığı
`-0.202429`, pooled yanlılık `-0.075725` olmuştur. Full-data seçiminde
`rho=0.00` çıkması Dixon-Coles düşük skor düzeltmesinin ek katkı sağlamadığını
gösterir; düzeltme OOS score NLL'yi bağımsız Elo-Poisson'a göre `+0.000403`
kötüleştirmiştir.

Karar `KEEP_SHADOW`dur. Production sözleşmesi, takım ratingleri ve sıralamalar
değişmemiştir. Skor matrisi `1e-10` tail toleransını gerçek lambda aralığında
korumak için adaptif olarak en fazla 25 gole genişletilir.

```text
output/scoreline_backtest_2018_2026/backtest_report.md
output/scoreline_backtest_2018_2026/selected_scoreline_model.json
```

### Turnuva ve Geçmiş-Sezon Gol Seviyesi Kalibrasyonu

Poisson gol beklentilerine aynı maç için ortak bir
`exp(c_competition+c_prior_season)` çarpanı uygulanarak göreli ev/deplasman güç
yönü korunmuştur. Turnuva/sezon strength gridleri ve `1/2/3` tamamlanmış sezon
penceresi toplam 65 aday üretmiştir. Her outer foldun seçimi yalnız geçmiş
sezonlardan oluşan inner walk-forward sonuçlarıyla yapılmıştır.

```text
Score NLL fold wins          2/6
1X2 Brier fold wins          2/6
1X2 log-loss fold wins       2/6
Pooled score NLL delta      +0.001527
Pooled Brier delta          +0.001086
Pooled log-loss delta       +0.001771
UCL goal bias               -0.179161
```

Turnuva offset'i hiçbir outer foldda seçilmemiştir. Geçmiş sezon offset'i dört
foldda seçilmesine rağmen yalnız iki foldda score NLL iyileştirmiştir. UCL gol
yanlılığı bir miktar azalmış, fakat UEL score NLL conservative güven aralığı
`[+0.000170,+0.004507]` ile güvenilir zarar göstermiştir. Pooled gol bias
`-0.051879` ile önceden belirlenen `0.05` sınırını da az farkla geçememiştir.

Karar `KEEP_SHADOW`dur. Bu düzeltmeler production'a, ratinglere veya mevcut AO
1X2 çıktısına eklenmez.

```text
output/scoreline_level_calibration_2018_2026/backtest_report.md
output/scoreline_level_calibration_2018_2026/selected_level_model.json
```

## Sabit Turnuva İlerleme Bonusu

Kazanan takıma eşleşme kesinleşince eklenen, kaybedenden puan düşmeyen ve yeni
sezonda sıfırlanan sabit turnuva bonusu ayrı bir katman olarak test edilmiştir.
AO First Elo, Power Elo ve kontrollü gol farkı çekirdeği değiştirilmemiştir.

Önceden belirlenen aday:

```text
UCL etap bonusu       +12   sezonluk UCL cap 48
UEL etap bonusu        +8   sezonluk UEL cap 32
UECL etap bonusu       +4   sezonluk UECL cap 16
```

Bonus yalnız son 16, çeyrek final, yarı final ve final eşleşmeleri tamamen
sonuçlandığında uygulanır. Knockout play-off, lig aşaması, ilk sekiz ve ön
elemeler kapsam dışıdır. Maç Power Elo güncellemesi sıfır-toplam kalırken bonus
bilinçli olarak non-zero-sum'dır; kullanıcıya yine tek `AO Live Elo = Power Elo
+ Turnuva Bonusu` gösterilmesi hedeflenir.

2018/19-2025/26 dönemindeki 4.884 unseen maçta `12/8/4` adayı:

```text
Brier fold wins          4/6
Log-loss fold wins       4/6
Pooled Brier delta      -0.000006941
Pooled log-loss delta   -0.000011253
Ranking-safe folds       4/5
Maximum team bonus       60
```

Pooled sonuçlar küçük bir iyileşme yönündedir ve güvenilir zarar yoktur; ancak
dependency güven aralıkları sıfırı keser. UEL segmenti Brier'da `+0.000029`,
log-loss'ta `+0.000035` gerilemiştir. Eğitim foldlarından otomatik seçilen
`16-20` ağırlıklı yüksek bonuslar unseen sonuçta kötüleştiği için nested seçim
`KEEP_DISABLED` kalmıştır. Önceden belirlenen konservatif `12/8/4`, güvenilir
zarar göstermemesi ve açık ürün onayıyla `PROMOTE_MANUAL` olarak production'a
alınmıştır. İstatistiksel etki küçük olduğundan 2026/27 prospective ledger'da
ayrıca izlenecektir.

```text
output/fixed_tournament_bonus_backtest_2018_2026/backtest_report.md
output/fixed_tournament_bonus_backtest_2018_2026/decision.json
output/progression_bonus_production_verification_2025_26/verification.json
```

### Aşama Ağırlıklı Progression Sonucu

KPO'yu kaldıran dört aşamalı Gentle, Linear, Final Heavy ve diagnostik Equal
Four profilleri, UCL toplam cap `30/45/60/75/90` ve sabit `3:2:1` turnuva
oranlarıyla aynı production çekirdeği üzerinde test edilmiştir. Dört aşamalı
adaylarda KPO bonus olayı yoktur; penaltıyla tur geçen takım tam bonus alır,
bonus sezon başında sıfırlanır ve Power Elo sıfır-toplam kalır.

```text
Nested seçim                 4 fold current, 2 fold Gentle cap 30
Pooled Brier farkı          -0.000017673
Pooled log-loss farkı       -0.000022539
Aynı-sezon ortak rank win    0/6
Forward ortak rank win       0/5
Tam geçmiş seçimi            CURRENT_FIXED_12_8_4_X5
Karar                        KEEP_CURRENT_PRODUCTION
```

Loss farkları küçük olumlu yöndedir, fakat tie/team/month dependency güven
aralıkları sıfırı keser. Pooled UCL/UEL/UECL ranking sonuçları gerileme
göstermiş ve forward-ranking kapıları geçmemiştir. Bu nedenle production
`12/8/4 × 5` sözleşmesi korunmuş; yeni profil otomatik aktive edilmemiştir.

KPO'yu koruyan beş aşamalı ikinci deneyde toplam cap sabit `60/40/20`
tutulmuş; yalnız aşama dağılımı değiştirilmiştir. Gentle, Linear ve Late Heavy
profillerinin hiçbiri altı eğitim foldunun hiçbirinde Equal Five production
profilini geçememiştir. Sabit OOS yüzeyinde üç aday da Brier ve log-loss'u çok
küçük kötüleştirmiş, UECL log-loss'ta üçü için de güvenilir zarar oluşmuştur.

```text
Nested seçim                 6/6 CURRENT_FIXED_12_8_4_X5
Gentle Brier / log farkı    +0.000004480 / +0.000007373
Linear Brier / log farkı    +0.000006172 / +0.000010120
Late Heavy Brier / log      +0.000007044 / +0.000011491
Tam geçmiş seçimi            CURRENT_FIXED_12_8_4_X5
Karar                        KEEP_CURRENT_PRODUCTION
```

```text
output/stage_weighted_progression_backtest_2018_2026/backtest_report.md
output/stage_weighted_progression_backtest_2018_2026/selected_candidate.json
output/five_stage_weighted_progression_backtest_2018_2026/backtest_report.md
output/five_stage_weighted_progression_backtest_2018_2026/selected_candidate.json
```

## 2025/26 Tam Sezon Replay Durumu

2025/26 sezonundaki 961 benzersiz maç ve 236 başlangıç takımı kesin UTC ve
`match_id` sırasıyla replay edilmiştir. Her tahmin sonuç görülmeden kaydedilmiş,
ardından saha skoru Power Elo güncellemesine uygulanmıştır. Çalışma production
ve shadow kararlarını değiştirmez.

### Historical Locked OOS

Yalnız 2024/25 sonuna kadar seçilmiş fold-6 parametreleri kullanılmıştır:

```text
Scale / H / K                  835.561497 / 148.544266 / 103.980986
Gol farkı                     alpha=0, tau=300
Historical Locked Brier       0.587906
Historical Locked log-loss    0.987690
Historical 6/4/2 Brier farkı -0.000047
Historical 6/4/2 log farkı   -0.000066
```

`6/4/2` bonus yönü üç turnuva segmentinde zarar göstermemiş ve
`CONSISTENT_SHADOW_SIGNAL` almıştır. Bununla birlikte tie/team/month
konservatif güven aralıkları sıfırı kestiği için terfi kararı yoktur.

### Current Model Counterfactual

```text
PRODUCTION Brier / log-loss        0.586725 / 0.985323
NO_GD_CONTROL farkı               +0.000113 / +0.000142
GD_PRIOR_GRID farkı               -0.000115 / -0.000146
GD_EXTENDED farkı                 -0.000080 / -0.000118
FIXED_TOURNAMENT_BONUS farkı      -0.000094 / -0.000128
DOMESTIC_ANCHORED farkı           -0.007494 / -0.010916
```

Gol farkını kapatma kolu iki loss metriğinde de kötüleşmiş ve `HARM_SIGNAL`
almıştır. `12/8/4` fixed bonus tüm turnuva segmentlerinde yönü koruduğu için
`CONSISTENT_SHADOW_SIGNAL` almıştır. Daha güçlü gol farkı kolları ile domestic
anchor bazı segment guardrail'lerini geçemediği için `MIXED_OR_INCONCLUSIVE`
durumundadır. Bunlar retrospective sonuçlardır; production kanıtı değildir.

Skor yan katmanında `SCORELINE_LEVEL` exact-score NLL ve toplam gol MAE
yönünde bu sezonun en iyi counterfactual sonucunu vermiştir. FORMAT_P_ADVANCE
284 tamamlanmış eşleşmede identity olasılığını Brier'da
`0.207063`ten `0.205473`e indirmiştir. xG eki yalnız ortak 180 maçtır ve
production-grade veri olmadığı için shadow olarak kalır.

Teknik invariantlar geçmiştir:

```text
Maksimum maç pair zero-sum hatası   9.095e-13
Maksimum sezon Power toplam hatası  5.821e-11
Bonus cap aşımı                     0
Sayısal determinism                 16/16 dosya hash eşleşmesi
Test paketi                         423 passed
```

Ana raporlar:

```text
output/season_replay_2025_26/replay_report.md
output/season_replay_2025_26/replay_manifest.json
output/season_replay_2025_26/model_comparison.csv
output/season_replay_2025_26/AO_2025_26_Sezon_Replay_Raporu.pdf
```

## Takım Bazlı Home/Away Context

Global `H=148.544266` yerine doğrudan kalıcı bir takım bonusu yazılmamıştır.
Yeni araştırma katmanı, her takımın yalnız önceki tamamlanmış sezonlardaki
maç öncesi AO beklentisine göre iç saha ve deplasman residual'larını ayrı
tahmin eder. Az verili veya yeni takım etkisi global `H` prior'ına shrink olur;
nötr sahada düzeltme sıfırdır. Katman yalnız pre-match olasılığı değiştirir,
Power Elo ratingini ve maç sonrası güncellemeyi değiştirmez.

```text
H_effective = clip(H_global + HomeEffect_home - AwayEffect_away, 0, 300)
```

`3/5` sezon lookback, `0.75/1.00` decay, `6/10/15/20` shrinkage ve
`25/35/50/75/100/150` Elo etki sınırından oluşan 96 takım adayı global
kontrolle altı expanding outer foldda değerlendirilmiştir.

```text
Nested fold win             5/6
Pooled Brier farkı         -0.000884804
Pooled log-loss farkı      -0.001264891
UCL Brier / log farkı      -0.000586 / -0.000744
UEL Brier / log farkı      -0.002417 / -0.003456
UECL Brier / log farkı     -0.000030 / -0.000105
Tam geçmiş seçimi           w3 / decay .75 / shrink 15 / cap 50
Karar                       KEEP_SHADOW_CANDIDATE
```

Pooled conservative CI Brier için `[-0.002010,+0.000165]`, log-loss için
`[-0.002814,+0.000148]` olmuştur. Ayrıca seçilen foldlarda context cap'e
dayanma oranı yaklaşık `%39–46` aralığına çıkabilmektedir. Yön olumlu olsa da
istatistiksel belirsizlik ve sınır kullanımı nedeniyle production
`H=148.544266` korunur. Sonradan tüm OOS yüzeyinden seçilen en iyi sabit aday
yalnız retrospektif duyarlılık analizidir ve terfi kanıtı sayılmaz.

```text
output/team_venue_context_backtest_2018_2026/backtest_report.md
output/team_venue_context_backtest_2018_2026/selected_candidate.json
output/team_venue_context_backtest_2018_2026/profile_summary.csv
```

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
output/dynamic_k_backtest_2018_2026/backtest_report.md
output/progression_probability_2018_2026/calibration_report.md
output/progression_reserve_final_backtest_2018_2026/backtest_report.md
output/match_context_backtest_2018_2026/backtest_report.md
output/domestic_regression_diagnostics_2018_2026/diagnostic_report.md
output/rejected_feature_heterogeneity_2018_2026/audit_report.md
output/scoreline_backtest_2018_2026/backtest_report.md
output/scoreline_level_calibration_2018_2026/backtest_report.md
output/fixed_tournament_bonus_backtest_2018_2026/backtest_report.md
output/season_replay_2025_26/replay_report.md
output/season_replay_2025_26/AO_2025_26_Sezon_Replay_Raporu.pdf
```
