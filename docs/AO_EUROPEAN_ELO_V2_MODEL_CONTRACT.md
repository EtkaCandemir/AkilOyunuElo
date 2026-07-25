# AO European Elo v2 Teknik Model Sözleşmesi

Sürüm: `ao-european-elo-v2.0-dev-freeze`

Dondurma tarihi: 20 Temmuz 2026

Operasyonel sözleşme revizyonu: 23 Temmuz 2026

Durum: Kontrollü `0.10/300` gol farkı production'da aktiftir; 2026/27 lig
aşaması ve sonrası için prospective izleme yapılacaktır. Sezonun tamamı
untouched değildir.

## 1. Karar Özeti

AO European Elo v2 üç ayrı state üretir:

1. `AO First Elo`: Sezon öncesi statik başlangıç gücü.
2. `Power Elo`: Oynanan maçlarla sıfır toplamlı değişen güç.
3. `Achievement Reserve`: Tur/kupa başarısı için tasarlanan ayrı, sıfır toplamlı olmayan rezerv.

Canlı rating şu kimliktir:

```text
AO Live Elo = Power Elo + Achievement Reserve
```

Aktif dondurulmuş modelde `Achievement Reserve = 0` olduğu için AO Live Elo ile
Power Elo aynıdır. Katman kodda mevcuttur fakat kalibrasyon terfi kapısını
geçmediği için aktif değildir.

Seçilen kararlar:

| Katman | Karar | Aktif değer |
| --- | --- | ---: |
| 500-2000 referans ölçeği | Aktif | multiplier `3.7136066548` |
| Country upper tail | Terfi yok | beta `0` |
| European history upper tail | Terfi yok | beta `0` |
| Exposure tail | Terfi yok | beta `0` |
| Dynamic Power çekirdeği | Terfi | Scale `835.561497`, H `148.544266`, K `103.980986` |
| Season Power carry | Kapalı | `0.00` |
| Standart 1X2 çıktı | Terfi | draw-at-even `0.24`, shape `1.00` |
| Kontrollü gol farkı | Aktif | alpha `0.10`, tau `300`, GD cap `4` |
| Progression bonus | Kapalı | base `0` |
| European Achievement Reserve | Kapalı | base `0` |
| UCL/UEL/UECL normal maç K çarpanı | Kapalı | uygulanmaz |

Bu sürüm production kanıtı iddia etmez. Parametre seçimi 2018/19-2025/26
geliştirme verisinde yapılmıştır. 2026/27 qualifying maçları model freeze
tarihinden önce başladığı için sezonun tamamı untouched değildir. Prospective
holdout yalnızca lig aşaması ve sonrasındaki kickoff öncesi kilitlerden oluşur.

## 2. 500-2000 Ölçek Sözleşmesi

V1.1 referans maksimumu `903.92`, taban `500` alınmıştır:

```text
M = 1500 / (903.92 - 500)
M = 3.713606654783126

AO Elo v2 = 500 + M x (AO Elo v1.1 - 500)
```

Dönüşüm component seviyesinde uygulanır:

| Component | v1.1 | v2 |
| --- | ---: | ---: |
| Base Rating | 500 | 500 |
| Domestic League Component | 140 | 519.904932 |
| Domestic Achievement Component | 160 | 594.177065 |
| European Prior Max Boost | 420 | 1559.714795 |

Hiçbir hesap sonunda `min/max` ile rating kesilmez. Fakat aktif statik config'te
country/europe/exposure tail beta değerleri `0`, maximum effective exposure
`0.85` ve erişilebilir maksimum Domestic Achievement `1.08`dir. Bu sınırlar aynı
takımda doyduğunda AO First Elo'nun teorik maksimumu tam olarak `2000` olur.
Dolayısıyla `2000`, final clipping sınırı değildir ama aktif AO First Elo için
yapısal saturation sınırıdır. Research tail adayları ve Dynamic/Live Elo bu
değeri aşabilir; aktif statik model aşamaz.

Bu affine dönüşüm v1.1 takım sırasını birebir korur. Dinamik `Scale`, saha
avantajı ve `K` de aynı `M` ile büyütüldüğü için beklenen skorlar ve maç başına
göreli öğrenme davranışı yalnızca ölçek değişimi nedeniyle değişmez. Bu tercih
kullanıcıya daha okunabilir puan farkı verir; tek başına tahmin doğruluğu kanıtı
değildir.

## 3. Zaman ve Veri Sızıntısı Sözleşmesi

- Statik CSV'lerde `season`, rating'in üretildiği hedef sezondur.
- `t`, hedef sezondan önce tamamlanan son sezondur.
- `t_minus_4`, beş sezonluk pencerenin en eski sezonudur.
- Hedef sezon içinde oluşan veri AO First Elo'ya geri yazılmaz.
- Maç olayları tam UTC timestamp ile işlenir.
- Aynı timestamp'te deterministik sıra `match_id` artan sırasıdır.
- Kronoloji gerilemesi reddedilir.
- Penaltı atışı golleri saha skoruna eklenmez.
- Saha skoru 90 dakika veya uzatma oynandıysa 120 dakika sonundaki skordur.

## 4. Beş Sezon Ağırlıkları

Ülke puanı, kulüp Avrupa geçmişi, played exposure ve match exposure aynı
ağırlıkları kullanır:

```text
t_minus_4 = 0.07
t_minus_3 = 0.13
t_minus_2 = 0.20
t_minus_1 = 0.27
t         = 0.33
toplam    = 1.00
```

Bu ortak ağırlık ayrı sinyaller arasında metodolojik tutarlılık sağlar ve veri
miktarı sınırlıyken serbest parametre sayısını azaltır. Veri arttığında her
sinyal için ayrı ağırlıklar yeni bir nested walk-forward araştırma adayı olabilir.

## 5. Country Strength

```text
Weighted Country Score = sum(w_i x country_points_i)

u_country = log(1 + Weighted Country Score) / log(1 + 25)

Tail(u, beta) =
    u                         , u <= 1
    1 + beta x (u - 1)        , u > 1

Country Strength Norm = Tail(u_country, country_tail_beta)
League Strength = Country Strength Norm ^ 0.80
```

Aktif `country_tail_beta = 0` olduğu için benchmark üzerindeki değerler mevcut
hard-cap davranışını korur. Uncapped değer output'ta saklanır; sinyalin cap'e
ulaşıp ulaşmadığı ayrıca görülebilir.

## 6. Domestic Achievement

Lig pozisyonu biliniyorsa:

```text
League Percentile =
    (league_team_count - domestic_position) / (league_team_count - 1)

Percentile Score = 0.15 + 0.70 x League Percentile
```

Şampiyonluk davranışı:

```text
is_league_champion = true ise League Finish Score = 1.00
```

Pozisyon eksik olsa da şampiyonluk bayrağı geçerlidir. Pozisyon verilmiş ve
şampiyonluk bayrağı `true` ise pozisyon tam olarak `1` olmalıdır.

Pozisyon bilinmiyor ve takım şampiyon değilse:

```text
League Finish Score = 0.10
```

Kupa ve duble:

```text
Cup Base Score = 0.62                         , is_cup_winner = true
Cup Double Bonus = 0.08 x League Finish Score, league + cup birlikte kazanıldıysa

Domestic Achievement Uncapped =
    max(League Finish Score, Cup Base Score) + Cup Double Bonus

Domestic Achievement = min(1.10, Domestic Achievement Uncapped)
```

Şampiyon ile ikinci sıra arasındaki step bilinçli bir domain kararıdır. 20
takımlı lig örneğinde bu basamak v2 ölçeğinde yaklaşık `89` puan Domestic Prior
farkı doğurabilir. Bu süreksizlik validation veya hesaplama bug'ı değildir;
ileride değiştirilmesi ancak ayrı bir ranking-first backtest ile mümkündür.

## 7. Domestic Prior

```text
Achievement Scale = 0.40 + 0.60 x League Strength

Domestic Prior =
    500
    + 519.9049316696 x League Strength
    + 594.1770647653 x Domestic Achievement x Achievement Scale
```

Achievement Scale, yerel başarının güçlü ligde daha fazla rating üretmesini
sağlar. Zayıf ligde de yerel başarı katkısının yüzde 40'ı korunur.

## 8. European Prior

```text
Weighted European History = sum(w_i x club_points_i)

u_europe = log(1 + Weighted European History) / log(1 + 20)
European History Norm = Tail(u_europe, european_tail_beta)

European Prior = 500 + 1559.7147950089 x European History Norm
```

Aktif `european_tail_beta = 0` olduğu için benchmark üzerindeki kulüp geçmişi
mevcut hard-cap davranışını korur. `official_club_coefficient` ana formüle
girmez; denetim alanıdır.

## 9. European Exposure

```text
Weighted Season Exposure = sum(w_i x played_i)

Weighted Match Exposure =
    sum(w_i x min(matches_i / match_cap_i, 1))

European Exposure =
    0.60 x Weighted Season Exposure
    + 0.40 x Weighted Match Exposure
```

Ham exposure `e` için:

```text
Effective Exposure =
    e                                      , e <= 0.85
    0.85 + beta_exp x (e - 0.85)           , e > 0.85
```

Aktif `beta_exp = 0` olduğundan tam exposure'ın final karışımdaki değeri
`0.85`tir. Böylece çok güçlü Avrupa kanıtında bile Domestic Prior'ın yüzde 15'i
korunur.

## 10. Final AO First Elo

```text
AO First Elo =
    Domestic Prior
    + Effective Exposure x (European Prior - Domestic Prior)
```

İnvariantlar:

- Ham ve effective exposure `[0,1]` aralığındadır.
- Effective exposure ham exposure'ı aşamaz.
- Avrupa geçmişi olmayan açık sıfır satırında iki exposure da `0` olur.
- Sıfır exposure takımında `AO First Elo = Domestic Prior` olur.
- Final rating Domestic Prior ile European Prior arasındaki kapalı aralıktadır.
- Rating hesap sonunda 500 veya 2000'e kırpılmaz.
- Aktif beta `0/0/0` davranışında AO First Elo'nun ulaşılabilir yapısal üst
  sınırı `2000`dir.

Rating kaynak kategorisi ham exposure ile belirlenir:

| Ham exposure | `rating_source_type` |
| ---: | --- |
| `0` | Pure Domestic Projection |
| `0 < e < 0.75` | Mixed Domestic-European Estimate |
| `e >= 0.75` | European Evidence-Based Rating |

## 11. Saturation ve Tail Diagnostikleri

Statik output şu denetim sinyallerini taşır:

- Country ve European history için uncapped norm.
- Tail sonrası norm.
- Tail excess ve tail active bayrakları.
- Achievement uncapped score ve cap active bayrağı.
- Ham ve effective exposure, exposure tail excess/active.
- Takım başına `saturation_count`.

`5 x 5 x 4 = 100` tail/exposure kombinasyonu altı outer foldda test edilmiştir.
Son değerlendirmede hedef, aday rating'lerden tamamen bağımsız cross-fitted
ölçümdür. Her takım için saha etkisi o takımın maçları dışarıda bırakılarak;
rakip gücü ise doğrudan target-opponent eşleşmeleri dışarıda bırakılarak tahmin
edilir. Düzeltilmiş maç skoru maç sayısı güvenilirliğiyle `0.5` merkezine
shrink edilir. Hiçbir tail adayı bu hedefte katı ranking-first kapılarını
geçmemiştir. Karar `NO_PROMOTION`, aktif aday `c0_e0_x0` olmuştur.

Dinamik optional katmanlarda aynı sezon hedefi kullanılmaz; aksi halde daha
sert güncelleme aynı sonuçları rating'e daha fazla yazdığı için yapay avantaj
kazanır. Her sezon sonu rating yalnız takip eden sezonun bağımsız
schedule-adjusted performansına karşı ölçülür. 2026/27 untouched kaldığından
2020/21-2024/25 outer-test sezon sonları için beş forward ranking değerlendirmesi vardır.

Tarihsel v2 dağılım kontrolü:

```text
Takım-sezon                  1,887
500-2000 bandındaki oran    100.000%
Medyan                      1075.83
Yüzde 90                    1747.65
Minimum / maksimum          651.49 / 1996.09
Tam 500 veya 2000           0
```

UCL saturation-count ile expected-score MSE korelasyonu Spearman `+0.0671`, Pearson
`-0.0050` bulunmuştur. Bu geliştirme örneğinde cap'e yapışma sayısı daha yüksek
hatanın güçlü bir açıklaması değildir. Bu sonuç cap kaynaklı çözünürlük riskini
sonsuz biçimde reddetmez; yalnızca mevcut hipotezi destekleyen kanıt bulunmadığını
gösterir.

## 12. Normalize Beklenen Puan ve Power Elo

```text
E_home = 1 / (1 + 10 ^ -((Home Live - Away Live + H) / Scale))

Scale = 835.5614973262
H     = 148.5442661913   (nötr sahada 0)
K     = 103.9809863339
```

Saha sonucu:

```text
S_home = 1.0  , ev sahibi saha skorunda kazanırsa
S_home = 0.5  , saha skoru berabereyse
S_home = 0.0  , ev sahibi saha skorunda kaybederse
```

`E_home` ve kalıcı kolon adı `expected_home_score`, ev sahibi galibiyet
olasılığı değildir. Beklenen normalize maç puanıdır: sonuç hedefi `1/0.5/0`
olur. Bu ayrım, hem API hem CSV çıktısında
`expected_score_semantics = normalized match points: win=1, draw=0.5, loss=0`
olarak saklanır.

Aktif güncelleme:

```text
Delta = K x (S_home - E_home)

Home Power New = Home Power Old + Delta
Away Power New = Away Power Old - Delta
```

Her maçta iki takımın toplam Power Elo'su korunur. Turnuva adı normal maç K'sına
çarpan uygulamaz. UCL maçının tipik zorluğu rakip rating farkı üzerinden gelir.

Exact-date nested walk-forward sonucu:

```text
Outer fold wins                  6/6
Unseen expected-score MSE farkı -0.003955
Clustered %95 CI                [-0.005491, -0.002370]
UCL / UEL / UECL MSE farkı      -0.004570 / -0.005282 / -0.002631
Full-data start-end Spearman    0.937454
Maksimum mutlak hareket         475.018
Guardrail                       742.721
```

Bu eski expected-score MSE sonucu yalnız legacy diagnostiktir. Yeni doğrulama
gerçek üç sınıflı olasılıklarla yapılmıştır:

```text
P_draw = 0.24 x (4 x E_home x (1-E_home)) ^ 1.00
P_home = E_home - 0.5 x P_draw
P_away = 1 - E_home - 0.5 x P_draw
```

İnvariantlar:

- `P_home + P_draw + P_away = 1`.
- Her olasılık `[0,1]` aralığındadır.
- `P_home + 0.5 x P_draw = E_home`; Elo beklenen puan anlamı değişmez.
- Standart Brier üç sınıftaki kare hataların toplamıdır ve `[0,2]` aralığındadır.
- Standart log-loss gerçekleşen H/D/A sınıfına atanan olasılığı kullanır.

Dinamik çekirdeğin tuned static karşılaştırıcıya karşı unseen standart 1X2
Brier farkı `-0.008929`, en geniş dependency-bootstrap yüzde 95 zarfı
`[-0.012823, -0.004990]` ve fold sonucu `6/6`dır. UCL, UEL ve UECL
segmentlerinin hiçbirinde güvenilir zarar yoktur. Karar `CONFIRMED_1X2`dir.

## 13. Season Power Carry

Yeni sezon başlangıcı:

```text
Power Start New =
    (1 - carry) x Current AO First Elo
    + carry x Previous Season Power End

active carry = 0.00
```

Önceki state'te olmayan takım doğrudan güncel AO First Elo ile başlar. Carry
adayları `0/0.25/0.50/0.75/0.85/0.90/1.00` olarak test edilmiştir.

```text
Full-data carry adayı      0.85
Nested unseen fold wins    5/6
Overall 1X2 Brier farkı   -0.005895
Dependency %95 zarfı      [-0.009464, -0.002653]
```

Carry toplamda ve UCL/UEL segmentlerinde yararlı görünür; ranking ve hareket
guardrail'leri de geçer. Ancak kabul sözleşmesi no-carry çekirdeğini her unseen
foldda geçmesini zorunlu tutar. 2021/22 fold farkı `+0.000912` olduğu için yalnız
`5/6` kazanmıştır. Bu nedenle full-data optimumu production'a taşınmamış ve
aktif `carry=0` yapılmıştır:

```text
Power Start New = Current AO First Elo
```

## 14. Gol Farkı Katmanı

İlk araştırmada sınanan fakat production'a alınmayan aileler:

```text
LOG                    = 1 + weight x ln(goal_difference)
SQRT                   = 1 + weight x (sqrt(goal_difference) - 1)
FAVORITE_DAMPED_LOG    = 1 + weight x ln(goal_difference) x favorite_correction
G                      = min(goal_cap, selected_family)
Delta = K x G x (S - E)
```

Final `carry=0` ve standart 1X2 robustness koşusunda full-data aday
`SQRT, weight=1.00, cap=2.00` olmuştur. Nested unseen sonuç `3/6` Brier fold,
`2/5` forward-ranking gerilemesiz fold ve her iki forward ranking metriğinde
iyileşme `0/5`tir. Brier farkı `-0.000403`, dependency envelope
`[-0.000984, +0.000176]`dır. Bu eski aile kapalı kalmıştır.

Production'da bunun yerine ayrı test edilen kontrollü formül aktiftir:

```text
D = Home Live - Away Live + H
M_GD = 1 + alpha x ln(min(GD, 4)) x exp(-abs(D) / tau)
alpha = 0.10
tau = 300
Delta = K x (S - E) x M_GD
```

Beraberlik ve tek farklı sonuçta `M_GD=1`dir. Penaltı atışları saha skoruna
eklenmez; penaltıyla sonuçlanan saha beraberliğinde `S=0.5`, `GD=0` ve
`M_GD=1` kalır. Skor 90 dakika veya oynandıysa 120 dakika sonundaki saha
skorudur. `GD` 4'te tavanlanır. `abs(D)` büyüdükçe bonus üstel biçimde söner;
bu nedenle ağır favorinin büyük galibiyeti yakın güçteki iki takım arasındaki
aynı gol farkından daha az ek bilgi taşır.

Önceden belirlenen `0.10/300` adayı 4,884 tarihsel OOS maçta Brier'ı `6/6`
foldda iyileştirmiştir. ΔBrier `-0.000174`, Δlog-loss `-0.000250`, pooled
Spearman `+0.000626` ve pooled pairwise `+0.000094` olmuştur. Maksimum gözlenen
çarpan `1.138347`dir. Bu kanıt ve manuel model kararıyla kontrollü katman
production'a alınmıştır.

### 14.1 Turnuva Bazlı K Hiyerarşisi

UCL/UEL/UECL katsayısı takımın doğrudan gücü değil, ilgili maçtan öğrenme
hızıdır. Hiyerarşik adaylar test edilirken global K da `0.75/1.00/1.25/1.50`
ile birlikte yeniden ölçeklenmiştir; böylece düşük UEL/UECL katsayısı sırf
ortalama K'yı düşürdüğü için elenmez.

Full-data ranking-first aday `UCL=1.50, UEL=0.75, UECL=0.675` olmuştur. Ancak
unseen sonuç yalnız `1/6` Brier fold ve `1/5` forward-ranking gerilemesiz fold,
Brier farkı `+0.000783`, envelope `[-0.000655, +0.002253]`tir. Training'de
hiyerarşi seçilmesine rağmen geleceğe genellenmediği için production'da tüm
turnuva K çarpanları `1.00` kalır.

## 15. European Achievement Reserve

Araştırılan ayrı katman:

```text
Advance Reserve =
    Base x Competition x Stage x (1 - P_advance)

Trophy Reserve =
    Base x Competition x (1 - P_advance)
```

`P_advance`, eşleşmenin ilk maçından hemen önce dondurulur. Mevcut uygulamadaki
değer gerçek bir iki maçlı tur olasılık modeli değildir; saha avantajı olmadan
hesaplanan tek maçlık Elo expected-score proxy'sidir. Reserve yalnızca tur
kesinleştikten sonra eklenir ve sonraki maçlarda AO Live Elo'ya girer. Normal
Power güncellemesi sıfır toplamlı kalır. Katman Base `0` olduğu için bu proxy
aktif rating'i şu anda etkilemez; reserve yeniden araştırılmadan önce gerçek
tek/iki maçlı progression probability ile değiştirilmelidir.

Test grid'i:

- Base: `0/74.27/111.41/148.54/185.68/222.82`
- UCL: `1.00`
- UEL: `0.50/0.65/0.80`
- UECL: `0.25/0.45/0.60`
- Zorunlu sıra: `UCL > UEL > UECL`
- Altı stage profili
- Decay: `0.25/0.50/0.75/1.00`
- Reserve cap: `297.088532`
- Toplam aday: `961`

Final robustness koşusunda full-data ranking-first aday `Base=222.816`,
`UEL=0.80`, `UECL=0.60`, `LATE_BALANCED`, `decay=0.50` olmuştur. Bu adayın
unseen sonucu yalnız `2/6` Brier fold ve `1/5` forward-ranking gerilemesiz
folddur. Brier farkı `+0.001982`, dependency envelope
`[+0.000686, +0.003321]` ile güvenilir zarar gösterir. Training seçiminin
geleceğe taşınmaması ve maç/tur bilgisini iki kez sayma riski nedeniyle karar
`DISABLE_ACHIEVEMENT_RESERVE`dır:

```text
reserve_base = 0
reserve_decay = 0
progression reserve = 0
trophy reserve = 0
```

Belgelerdeki `UCL=1.00`, `UEL=0.65`, `UECL=0.45` değerleri kapalı bir araştırma
konfigürasyonunun hiyerarşisidir. Base sıfır olduğu için production rating'ine
puan eklemez.

## 16. Harici ClubElo Bulgusu ve Kalan Risk

Final benchmark; `carry=0`, fold'a özel geçmişten seçilmiş Dynamic Core,
training-only 1X2 draw mapping ve exact UTC kronoloji kullanır. Paired ClubElo
arşivi 363 unseen maç, bunun içinde 171 UCL maçı kapsar:

```text
UCL AO Dynamic 1X2 Brier   0.601615
UCL AO Static 1X2 Brier    0.606032
UCL ClubElo 1X2 Brier      0.572126
AO - ClubElo               +0.029489
Dependency envelope        [-0.002932, +0.064568]
```

Dynamic çekirdek AO Static'e karşı UCL'de `-0.004417` iyileşir. ClubElo nokta
tahmini daha iyi olsa da tie/match, team-season ve calendar-month görünümlerinin
conservative envelope'ı sıfırı kestiği için final modelde güvenilir üstünlük
ilan edilmez. Bu, farkın kapandığı anlamına da gelmez. ClubElo snapshot arşivi
ağırlıkla yerleşik kulüpleri içerir ve tüm qualifier evrenini temsil etmez.

UCL açığı en çok lig/ana aşama, son 16 ve AO expected-score `0.40-0.60`
bandında görülür. 500-2000 affine ölçek farkı tek başına gidermez; Scale da aynı
oranda büyütülür. Risk saklanmaz ve 2026/27 prospective lig-aşaması segment
raporunda ayrı izlenir.

## 17. Statik Input CSV Sözleşmesi

| Dosya | Anahtar | Rol |
| --- | --- | --- |
| `teams.csv` | `team_id` | Takım kimliği, ülke ve lig metadata'sı |
| `country_coefficients.csv` | `season + country_code` | Beş sezon ülke/lig puanı |
| `domestic_context.csv` | `season + team_id` | Lig pozisyonu, şampiyonluk, kupa, giriş metadata'sı |
| `club_european_points.csv` | `season + team_id + country_code` | Kulüp puanı ve exposure kanıtı |

Her takım için club history satırı zorunludur. Avrupa geçmişi yoksa beş sezon
puan, played ve matches açıkça sıfır yazılır. Eksik, negatif, NaN veya sonsuz
puanlar; duplicate anahtarlar; geçersiz boolean; pozisyon/takım sayısı çelişkisi;
pozitif olmayan match cap reddedilir.

`official_five_year_total`, `official_country_rank`,
`official_club_coefficient` ve `country_part` opsiyonel denetim alanlarıdır.

## 18. Dinamik Input ve Output Sözleşmesi

Prospective tahmin input'u sonuç içermez:

```text
fixtures.csv:
match_id, season, kickoff_utc, competition, round,
tie_id, is_knockout, is_tie_decider, stage,
home_team_id, away_team_id, is_neutral
```

`is_knockout` kolonu zorunludur. Değer `true` ise `tie_id` ve `stage` zorunlu,
turun son maçıysa `is_tie_decider=true` ve settle sonucunda
`advanced_team_id` zorunludur.

Settlement ve retrospective replay için `matches.csv` alanları:

```text
match_id, season, kickoff_utc, competition, round,
home_team_id, away_team_id, home_goals, away_goals,
is_neutral, decided_on_penalties
```

Eleme metadata'sı:

```text
tie_id, is_knockout, is_tie_decider, stage, advanced_team_id
```

Motor şunları reddeder:

- Duplicate `match_id`.
- Kronoloji gerilemesi.
- State'te bulunmayan takım.
- Negatif veya tam sayı olmayan skor.
- Geçersiz turnuva/tur.
- Aynı `tie_id`nin farklı takım, turnuva veya stage ile tekrar kullanılması.
- State ile config'in model version veya config fingerprint uyuşmazlığı.

Çıktılar:

| Dosya | İçerik |
| --- | --- |
| `ratings_state.csv` | İnsan tarafından okunabilir takım rating snapshot'ı |
| `state_checkpoint.json` | Processed maçlar, açık tie state'i, global kronoloji ve ratings checksum'u |
| `match_updates.csv` | Pre/post ratingler, normalize beklenti, H/D/A olasılıkları, saha skoru, G, Power Delta ve reserve eventleri |
| `replay_predictions.csv` | Sonuçlarla aynı batch'te üretilen, holdout olmayan retrospective audit |
| `batch_manifest.json` | Replay modunu ve `prospective_holdout_evidence=false` bilgisini taşır |
| `pre_match_log.csv` | Sonuçsuz fixture'dan kickoff öncesi üretilen append-only, hash-zincirli tahmin ledger'ı |

`ratings_state.csv` tek başına aynı sezon resume sözleşmesi değildir; processed
maç ve açık eşleşme bilgisi için checkpoint JSON ile birlikte kullanılmalıdır.
Prediction ledger'ın hash zinciri satır değişikliğini görünür kılar, fakat
harici güvenilir zaman damgasının yerini tutmaz.

Eşzamanlı farklı maçlar kickoff öncesi aynı state'ten kilitlenebilir. Settlement
`kickoff_utc`, ardından `match_id` artan sırasıyla yapılır. Aynı kickoff'taki
önceki bağımsız settlement global state konumunu ilerletse de kilitli maçın home/
away pre-rating değerlerinden biri değişmişse settlement reddedilir.

Tam kolon sırası makinece okunabilir sözleşmede tutulur:
`contracts/ao_european_elo_v2.json`.

## 19. Public Python API

```python
expected_score(...)
expected_1x2_probabilities(...)
initialize_season(...)
update_match(...)
apply_progression(...)
run_season(...)
lock_prediction(...)
settle_locked_match(...)
```

Tek maç kernel'i input state'i mutate etmez. Aynı başlangıç state'i, aynı config
ve aynı input aynı sonucu üretir. CSV batch CLI aynı Python kernel'ini kullanır.

```bash
python3 scripts/run_dynamic_elo.py \
  --initial-ratings initial_ratings.csv \
  --matches matches.csv \
  --output-dir output/replay_run
```

Bu batch komutu yalnızca retrospective replay'dir. Prospective akış
`run_dynamic_live.py initialize`, `lock` ve `settle` komutlarıyla yürütülür.

## 20. Pilot ve Regresyon Sözleşmesi

V1.1 API ve sayısal pilot değerleri korunur. V2 beta=0 rating'i v1.1'in tam
affine dönüşümüdür ve sıralama değişmez.

Gerçek 10 takım pilotu:

| Sıra | Takım | AO First Elo v2 |
| ---: | --- | ---: |
| 1 | Arsenal | 1992.870 |
| 2 | Sporting CP | 1926.711 |
| 3 | Benfica | 1881.335 |
| 4 | Shakhtar Donetsk | 1764.684 |
| 5 | Galatasaray | 1756.282 |
| 6 | AZ Alkmaar | 1741.804 |
| 7 | Slavia Praha | 1621.856 |
| 8 | Pafos | 1444.036 |
| 9 | Como | 1421.436 |
| 10 | Omonia Nicosia | 1398.415 |

Como sıfır exposure kontrolüdür: final rating'i doğrudan v2 Domestic Prior'dır.
Bu sürümde Arsenal'in üstünde değildir ve gerçek pilot v1.1 sırası korunur.

## 21. 2026/27 Prospective Lig Aşaması Holdout'u

Holdout başlamadan dondurulanlar:

- Model version ve config fingerprint.
- Statik ve dinamik bütün aktif parametreler.
- Kapalı optional katmanların sıfır değerleri.
- Fixture/result ayrımı, checkpoint, update ve pre-match ledger sözleşmeleri.
- Saha skoru, penaltı ve kronoloji kuralları.
- H/D/A dönüşümü, standart Brier ve standart log-loss tanımları.

Holdout sırasında yasak:

- Sonuca bakarak Scale, H, K veya carry değiştirmek.
- UCL segmentini düzeltmek için ara parametre eklemek.
- Goal margin veya reserve katmanını açmak.
- Geçmiş tahmin loglarını yeniden üretip eski kaydın üzerine yazmak.

Kapsam:

- En erken prospective başlangıç `2026-09-08T00:00:00Z`.
- UCL, UEL ve UECL lig aşaması ve sonrası.
- Qualifying ve play-off maçları kapsam dışı.
- Yalnızca `generated_at_utc < kickoff_utc` olan ledger satırları dahil.
- Retrospective replay satırları hiçbir koşulda holdout kanıtı sayılmaz.

İlk anlamlı ara rapor 2027'de lig aşamaları bittikten sonra; nihai rapor Avrupa
sezonu tamamlandıktan sonra hazırlanır. Ana olasılık metrikleri standart 1X2
Brier/log-loss'tur; legacy expected-score MSE yalnız karşılaştırma diagnostikidir.
Spearman, pairwise ranking, maksimum rating hareketi ve
UCL/UEL/UECL segmentleri birlikte sunulur. 2027/28 bir sonraki tam sezon holdout
adayıdır. Ayrıntılı operasyon sözleşmesi `docs/HOLDOUT_PROTOCOL_2026_27.md`
dosyasındadır.

## 22. Bilinen Metodolojik Borçlar

- Altı outer fold güçlü ama sınırlı bir örneklemdir. Fold sayısı tek başına
  bağımsız tekrar sayısı gibi yorumlanmamalıdır.
- Tie/match, team-season ve calendar-month bootstrap sonuçları birer bağımlılık
  duyarlılık analizidir; formal multi-way clustered standart hata iddiası değildir.
- Draw modeli tek global parametre çiftiyle aktiftir. Veri büyüdüğünde stage veya
  turnuva bazlı ayrım yalnız nested walk-forward kanıtıyla araştırılmalıdır.
- Achievement Reserve tekrar açılmadan önce `P_advance` gerçek tek/iki maçlı
  tur olasılığı olarak kalibre edilmelidir.
- Şampiyonluk step'i bilinçli domain kararıdır; yalnızca ranking-first backtest
  ile değiştirilebilir.
