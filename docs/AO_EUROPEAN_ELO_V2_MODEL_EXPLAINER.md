# AO European Elo v2 — Ayrıntılı Model Anlatımı

Bu belge modelin çalışma biçimini uçtan uca anlatır. Sayısal değerler
`contracts/ao_european_elo_v2_production.json` ve `src/ao_elo/` kaynak kodundan
doğrulanmıştır; eski README, PDF veya rapor değerleri otorite kabul edilmemiştir.

**Otorite sırası:** contract JSON > `src/ao_elo` kaynak kodu >
`reports/current_model/active_model_snapshot.json` >
`reports/current_model/current_model_evaluation_report.md` > MODEL_STATUS /
README / eski raporlar / PDF.

---

## 1. Yönetici özeti

AO European Elo, UEFA kulüp turnuvalarında iki ayrı soruyu cevaplar. Birincisi
"bu kulüp şu anda ne kadar güçlü?", ikincisi "bu maçın sonucu ne olur?". Sistem
bu iki soruyu **ayrı katmanlarda** çözer ve bu ayrım belgenin tamamında korunur.
Rating katmanı kulüp gücünü üretir; prediction katmanı o gücü girdi olarak alıp
olasılığa çevirir. Prediction katmanı rating'e geri beslenmez.

Sezon başında her kulübün bir başlangıç gücü olmalıdır, ama Avrupa'da yeni
oynayan bir kulübün Avrupa geçmişi yoktur. Model bu boşluğu **AO First Elo** ile
doldurur: kulübün kendi ülkesindeki lig gücü, geçen sezonki lig sırası ve kupa
başarısı bir "yerel tahmin" üretir; kulübün Avrupa'daki geçmiş performansı ayrı
bir "Avrupa tahmini" üretir; ve ikisi kulübün Avrupa'ya ne kadar maruz kaldığına
göre karıştırılır. Hiç Avrupa oynamamış bir kulüp tamamen yerel tahminle başlar.

Bu karışımın bir üst sınırı vardır. Kulüp ne kadar çok Avrupa oynamış olursa
olsun, yerel kanıtın ağırlığı `0.35`'in altına inmez. Sebep istatistiksel
değil, yapısaldır: Avrupa maç sayısı kulübün gücünün yanı sıra ne kadar ileri
gittiğinin de fonksiyonudur, ve o iki şeyi birbirinden ayırmak için elde yeterli
kanıt yoktur.

Sezon başladıktan sonra **Power Elo** devreye girer. Her maçta iki takım
arasında sıfır toplamlı bir puan alışverişi olur: kazanan ne kadar alırsa,
kaybeden o kadar kaybeder. Beklenmedik sonuçlar daha çok puan hareket ettirir.
Gol farkı bu hareketi büyütür, ama sınırlıdır ve favorinin farklı kazanması daha
az ödüllendirilir. xG varsa performansın skorla ne kadar örtüştüğüne bakılıp
küçük bir düzeltme yapılır; xG yoksa bu adım hiç uygulanmaz ve sıfır sayılmaz.

Eleme turları ayrı ele alınır. Bir ön eleme maçı finalle aynı ağırlıkta olamaz,
ama hiç saymamak da yanlış olur. Model her eleme turuna bir çarpan verir
(`Q1` için `0.20`'den `MAIN` için `1.00`'e). Ana tura geçişte hiçbir sıfırlama
veya taşıma yapılmaz — kulübün ratingi yalnız maç oynadığında değişir.

**AO Live Elo**, Power Elo'nun üstüne iki şey ekler: kapalı olan Achievement
Reserve ve aktif olan European Progression Bonus. Progression Bonus yalnız son
16'dan itibaren, yalnız turu geçene, tur başına bir kez verilir ve kaybedenden
düşülmez — yani sıfır toplamlı değildir. Sezon sonunda sıfırlanır.

Olasılık tarafında üç kaynak vardır. **Current AO 1X2**, ratingi doğrudan
H/D/A'ya çevirir. **Structural Logistic**, kickoff'tan önce bilinen özelliklerle
eğitilmiş bir lojistik regresyondur. **Domestic Poisson**, kulüplerin kendi
liglerindeki hücum/savunma parametrelerinden iki gol beklentisi üretip skor
matrisinden H/D/A çıkarır. Üçü log-olasılık uzayında birleştirilir.

Nihai servis edilen olasılık `%50` Current ML ve `%50` AO Domestic Poisson'ın
birleşimidir. İç blend'ler açıldığında efektif katkılar `%30` Current AO,
`%45` Structural ML ve `%25` Poisson olur. Bileşenlerden biri çalışmazsa satır
sessizce bozulmaz: `FALLBACK_CURRENT_AO` durumuna düşer ve sebebi loglanır.

Modelin ölçülmüş kazancı küçüktür. Geliştirme penceresinde ensemble, Current
AO'ya göre Brier'de `-0.003362` iyileşme sağlar. Bu bir geliştirme dönemi
walk-forward ölçümüdür ve 2026/27 prospective kanıtı **değildir**. Otomatik
kapı ensemble'ı `KEEP_SHADOW` olarak işaretler; production'da aktif olması bir
ürün kararıdır, otomatik bir terfi değildir.

Son olarak: bu repository canlı bir ingestion servisi değildir. Fetch worker,
kalıcı veritabanı, identity servisi, prediction lock kuyruğu, alarmlar ve read
API burada mevcut değildir. Bu belge modelin ne yaptığını anlatır; canlı
işletimin ne gerektirdiği §19'dadır.

---

## 2. Aktif sürüm ve otorite

| Alan | Değer |
| --- | --- |
| Rating model version | `ao-european-elo-v2.0-dev-freeze` |
| Production revision | `2026-08-28-domestic-provider-season-and-fixture-integrity-fixes` |
| Revision türü | `BUGFIX_NO_PARAMETER_RETUNING` |
| Prediction model version | `ao-ml-poisson-ensemble-v1-production` |
| Prediction kararı | `PROMOTE_WITH_MONITORING` |
| Prediction aktivasyon tarihi | `2026-08-13` |
| Contract | `contracts/ao_european_elo_v2_production.json` |
| Artifact manifest | `artifacts/production_prediction/manifest.json` |
| Manifest SHA-256 | `d577e9a883351a534e1be17a081f83f1248b65b2eb9791b994744832ea5bd021` |
| Aktif snapshot | `reports/current_model/active_model_snapshot.json` |

`revision_kind = BUGFIX_NO_PARAMETER_RETUNING` önemlidir: bu revision veri ve
giriş doğrulama kusurlarını kapatır, hiçbir model parametresini yeniden
seçmez.

---

## 3. Uçtan uca veri ve hesap akışı

```text
ham sezon verisi
  -> kimlik cozumlemesi (permanent club_id)
  -> domestic / european statik ozellikler
  -> AO First Elo                       (sezon basi guc)
  -> Power Elo mac replay'i             (mac mac guncelleme)
  -> Achievement Reserve + Progression Bonus
  -> AO Live Elo                        (servis edilen guc)
  -> Current AO 1X2                     (ratingten olasilik)
  -> Structural ML  ve  Domestic Poisson
  -> servis edilen ensemble
  -> prediction log
```

Her adım bir öncekinin çıktısını girdi olarak alır; hiçbir adım kendisinden
sonraki bir adımın çıktısını okumaz. Prediction katmanı rating state'ini
değiştirmez (`prediction_layer.rating_feedback = false`).

### Aynı kickoff UTC batch'i ve sızıntı koruması

Aynı anda başlayan maçlar **tek bir pre-match snapshot** ile işlenir. Eğer aynı
kickoff'taki maçlar sırayla işlenseydi, ikinci maç birinci maçın sonucundan
güncellenmiş ratingi görürdü — oysa gerçek hayatta ikisi de aynı anda
oynanıyordu. Bu yüzden batch içindeki tüm maçlar aynı rating durumundan okur,
güncellemeler batch sonunda uygulanır.

Domestic tarafta aynı kuralın karşılığı contract'ta şöyle yazar:
`domestic_chronology = STRICTLY_INCREASING_KICKOFF_PER_LEAGUE_COMPLETE_BATCH`
ve `same_team_same_kickoff = REJECT`. Bir lig içinde geriye tarihli batch veya
aynı takımın aynı kickoff'ta iki maçı reddedilir; geçersiz batch state
değiştirilmeden geri çevrilir (`domestic_invalid_batch = REJECT_BEFORE_STATE_MUTATION`).

Prediction tarafında cutoff kuralı:
`domestic_prediction_cutoff = LEAGUE_LAST_KICKOFF_LE_GENERATED_AT_AND_LT_FIXTURE_KICKOFF`.
Yani kullanılan domestic state'in son maçı hem üretim anından önce hem hedef
fikstürün kickoff'undan önce olmalıdır.

---

## 4. AO First Elo

AO First Elo, sezon başındaki kulüp gücüdür. İki ayrı tahminin ağırlıklı
karışımıdır.

### 4.1 Domestic Prior

Kulübün kendi ülkesindeki kanıttan üretilen tahmin.

Önce ülke gücü normalize edilir:

```text
uncapped_norm  = ln(1 + weighted_country_score) / ln(1 + country_strength_benchmark)
League Strength = uncapped_norm ^ gamma
```

`weighted_country_score`, son beş sezonun UEFA ülke puanlarının
`0.07 / 0.13 / 0.20 / 0.27 / 0.33` ağırlıklarıyla toplamıdır (en eskiden en
yeniye). `country_strength_benchmark = 25.0`, `gamma = 0.8`,
`country_tail_beta = 0.0` — yani üst kuyruk sönümlemesi kapalıdır.
(`src/ao_elo/features.py:55`, `src/ao_elo/scoring.py:8`)

Sonra yerel başarı ölçeklenip eklenir:

```text
Achievement Scale = achievement_alpha + (1 - achievement_alpha) * League Strength
                  = 0.40 + 0.60 * League Strength

Domestic Prior = base_rating
               + domestic_league_component * League Strength
               + domestic_achievement_component * Domestic Achievement * Achievement Scale

               = 500
               + 519.9049316696377 * League Strength
               + 594.1770647653002 * Domestic Achievement * Achievement Scale
```

(`src/ao_elo/scoring.py:28`)

`Achievement Scale` aynı yerel başarının güçlü bir ligde daha fazla kanıt
taşımasını sağlar: zayıf ligde şampiyon olmak, güçlü ligde şampiyon olmakla aynı
şey değildir.

### 4.2 Domestic Achievement ve lig sırası

Bilinen lig sırası doğrudan yüzdeliğe çevrilir:

```text
Position Percentile = (league_team_count - domestic_position) / (league_team_count - 1)

League Finish Score = percentile_floor + percentile_scale * Position Percentile ^ percentile_delta
                    = 0.15 + 0.70 * Position Percentile ^ 1.0
```

`percentile_floor = 0.15`, `percentile_scale = 0.7`, `percentile_delta = 1.0`.
Şampiyonluk bilgisi varsa taban `champion_base_score = 1.00`'a yükselir:
`League Finish Score = max(percentile_score, champion_base)`.
(`src/ao_elo/features.py:68`)

### 4.3 Kupa katkısı

Kupa şampiyonluğunun tabanı `cup_base_score = 0.62`'dir. Aktif kural:

```text
Domestic Achievement = min(achievement_cap, max(L, C) + w * min(L, C))

L = League Finish Score
C = cup_base_score   (kupa kazanmadiysa 0)
w = cup_contribution_weight = 0.12903225806451613
achievement_cap = 1.10
```

(`src/ao_elo/features.py:217`, contract `domestic_cup_contribution`)

Ağırlık türetilmiştir, seçilmemiştir:

```text
w = cup_double_bonus_multiplier * champion_base_score / cup_base_score
  = 0.08 * 1.00 / 0.62
  = 0.12903225806451613
```

(`src/ao_elo/features.py:198`)

Bu türetmenin amacı, önceki kuralın zaten ödüllendirdiği grubu **birebir**
korumaktır: şampiyon + kupa toplamı iki kuralda da `1.0800`'dir. Değişiklik saf
biçimde "aynı kuralı şampiyon olmayanlara da uygula" demektir.

**Neden `max` tek başına yetmiyordu:** `max(L, C)` kupayı bir *taban* yapar. Lig
skoru `0.62`'nin üstünde olan bir kupa şampiyonu kupasından **sıfır** kredi alır,
ve etki ters yönlüdür — ligde ne kadar kötüysen kupa o kadar değerli olur.
Katkı terimi bu boşluğu kapatır.

### 4.4 Bilinmeyen lig sırası

Lig sırası bilinmiyorsa:

```text
League Finish Score = max(unknown_league_finish_score, champion_base if champion else 0)
unknown_league_finish_score = 0.15
```

`0.15` percentile eğrisinin tabanıdır. Önceki değer `0.10` idi ve kanıt
yokluğunu **son sırada bitirmekten ağır** cezalandırıyordu; contract bunu
`UNKNOWN_IS_NEVER_BELOW_THE_PERCENTILE_FLOOR` kuralıyla kayda geçirir.
2018-2026 penceresinde bu düzeltmeden etkilenen takım-sezonu sayısı `1`'dir
(contract `unknown_domestic_position.affected_team_seasons_2018_2026`).

### 4.5 European Prior ve katılım normalizasyonu

Kulübün Avrupa geçmişinden üretilen tahmin:

```text
European Prior = base_rating + european_prior_max_boost * european_history_norm
               = 500 + 1559.714795008913 * european_history_norm
```

(`src/ao_elo/scoring.py:75`)

`european_history_norm`, ağırlıklı Avrupa geçmişinin
`european_history_benchmark = 20.0` ile normalize edilmiş hâlidir.

Katılım normalizasyonu bu girdiye uygulanır:

```text
rate = weighted_european_history * (1 + k) / (weighted_season_exposure + k)
k = european_participation_shrinkage = 0.20
```

(`src/ao_elo/scoring.py:46`, contract `european_participation`)

**Çözdüğü problem:** iki kulüp beş sezonda aynı toplam Avrupa puanını
toplamışsa ama biri beş sezonun beşinde, diğeri ikisinde oynamışsa, ikisinin
Avrupa gücü aynı değildir. Ham geçmiş toplamı bu farkı görmez. Normalizasyon
geçmişi, kulübün *girebildiği* sezonlar üzerinden yeniden ölçekler.

`(1 + k)` payı katmanın güvenliğini sağlar: tam katılımda `pw = 1` olduğundan
`rate = history * (1+k)/(1+k) = history` olur, yani değer **birebir korunur**.

### 4.6 European exposure ve blend

```text
european_exposure = exposure_season_weight * season_component
                  + exposure_match_weight  * match_component
                  = 0.60 * season + 0.40 * match

effective_exposure = min(european_exposure, max_european_exposure)
                   = min(european_exposure, 0.65)

AO First Elo = Domestic Prior + effective_exposure * (European Prior - Domestic Prior)
```

(`src/ao_elo/scoring.py:83`, `src/ao_elo/scoring.py:92`)

`exposure_tail_beta = 0.0` olduğu için cap üstünde ek bir kuyruk yoktur; cap
sert bir tavandır. Bunun sonucu `minimum_domestic_prior_weight = 0.35`: yerel
kanıtın ağırlığı hiçbir kulüpte `0.35`'in altına inmez.

### 4.7 Domestic Surprise

Kulübün kendi lig sırasının beş sezonluk dağılımına göre sürpriz yapıp
yapmadığını ölçer.

```text
normalized_volatility = min(1, 2 * weighted_volatility)
consistency           = 1 - variance_penalty * normalized_volatility
                      = 1 - 0.50 * normalized_volatility

domestic adjustment   = coefficient * consistency * (current_finish - weighted_expected_finish)
                      (theta = 0.40, cap +/-30)

AO adjustment = (1 - effective_european_exposure) * domestic adjustment
```

(contract `domestic_surprise`, `src/ao_elo/domestic_surprise_variance.py:77`)

Beş sezondan az geçmişi olan kulüpte hiçbir düzeltme uygulanmaz
(`minimum_history_seasons = 5`, `insufficient_history_behavior = NO_ADJUSTMENT`).
Exposure çarpanı katmanın amacını korur: Avrupa kanıtı arttıkça yerel sürprizin
ağırlığı azalır.

### 4.8 Cap, floor ve invariantlar

Aşağıdakiler kodda ve testlerde pinlidir:

| İnvariant | Neden |
| --- | --- |
| Tam katılımda normalizasyon nötrdür | `(1+k)/(1+k) = 1`; beş sezonun beşinde oynayan kulüp hareket etmez |
| Hiç katılmamış kulüpte oran sıfırdır | `played_weight <= 0` ise `rate = 0`; o satırlar `european_exposure = 0` taşıdığı için prior blend tarafından zaten yok sayılır |
| Katılım normalizasyonu rating'i düşürmez | Payda `pw + k <= 1 + k` olduğundan oran ham geçmişten küçük olamaz |
| Kupa kazanmayanda katkı sıfırdır | `min(L, C) = min(L, 0) = 0` |
| Kupa katkısı bir taban değildir | Kural `max(L,C) + w*min(L,C)`; `max` hâlâ tabanı verir, katkı onun üstüne eklenir |
| `league_finish_score` floor altına düşmez | Bilinen sırada `percentile_floor = 0.15` taban; bilinmeyende `unknown_league_finish_score = 0.15` |
| `achievement_cap` bir korkuluktur | `1.10` bir güvenlik tavanıdır; **erişilebilir maksimum değildir**. Aktif ağırlıkta ulaşılabilir en yüksek değer `1.0800`'dir, yani cap bu parametrelerle hiç bağlamaz |

Son satır önemlidir: `achievement_cap = 1.10` görüldüğünde "en yüksek başarı
`1.10`'dur" sonucu çıkarılmamalıdır. Cap yalnız beklenmedik bir parametre
kombinasyonuna karşı korkuluktur.

---

## 5. Match Power Elo

### 5.1 Beklenen sonuç

```text
E_home = 1 / (1 + 10 ^ ( -(R_home - R_away + H) / S ))

S = elo_scale      = 835.5614973262034
H = home_advantage = 148.54426619132505   (notr sahada 0)
```

(`src/ao_elo/dynamic.py:752`, contract `dynamic_core`)

### 5.2 Güncelleme ve sıfır toplam

```text
Delta_base = K * (S_home - E_home)

K = k_factor = 103.98098633392752
S_home = 1.0 galibiyet, 0.5 beraberlik, 0.0 maglubiyet

R_home' = R_home + Delta
R_away' = R_away - Delta
```

Ev sahibine eklenen miktar deplasmandan **birebir** düşülür. Bunun sebebi
`E_home + E_away = 1` özdeşliğidir: iki takımın beklentisi toplamda 1 olduğu
için, gerçekleşen sonucun beklentiden sapması iki takım için eşit büyüklükte ve
ters işaretlidir. Toplam rating kütlesi maç başına korunur. Gol farkı ve xG
düzeltmeleri de aynı `Delta` üzerinde çarpan/toplam olarak çalıştığı için
sıfır toplam bozulmaz (contract `xg_performance.zero_sum = true`).

### 5.3 Qualification stage-K

| Tur | Base importance | Retention | Effective K çarpanı |
| --- | ---: | ---: | ---: |
| `Q1` | `0.40` | `0.50` | `0.200` |
| `Q2` | `0.55` | `0.50` | `0.275` |
| `Q3` | `0.70` | `0.50` | `0.350` |
| `QUALIFYING_PLAYOFF` | `0.85` | `0.50` | `0.425` |
| `MAIN` | `1.00` | — | `1.000` |

```text
effective_K = base_k * stage_k_multiplier(round)
```

(`src/ao_elo/qualification_stage_k.py:90`, contract `qualification_transition`)

`Preliminary Round`, `Q1` olarak ele alınır. Çarpan **her qualifier maçına
gömülüdür** (`application = EMBEDDED_IN_EACH_QUALIFIER_MATCH`), sonradan
uygulanan bir düzeltme değildir.

**MAIN girişinde ne olmaz:** reset yok, carry yok, maç dışı rating değişimi yok
(`carry_formula = NO_MAIN_ENTRY_RESET`, `non_match_rating_change = false`).
Kulübün ratingi yalnız maç oynadığında değişir. Doğrudan ana tura giren kulüp
için de bir taşıma uygulanmaz (`direct_entrant_behavior = NO_CARRY`). Kulüp
kimliği kupalar arasında süreklidir (`cross_competition_state = CONTINUOUS_CLUB_ID_NO_RESET`);
UEL'e düşen bir kulüp ratingini yanında taşır.

Eleme turlarında progression bonusu verilmez
(`progression_bonus_in_qualifying = false`).

---

## 6. Goal margin ve xG

### 6.1 Uygulama sırası

```text
Delta = Delta_base * GD_multiplier + Delta_xG
```

Sıra bağlayıcıdır: önce temel Elo deltası, sonra gol farkı çarpanı, en son xG
düzeltmesi (contract `xg_performance.final_formula`).

### 6.2 Gol farkı çarpanı

```text
GD_multiplier = 1 + alpha * ln(min(GD, goal_difference_cap)) * exp(-|D| / tau)

alpha = 0.15
tau   = 300.0
goal_difference_cap = 4
D = efektif rating farki
```

(contract `goal_margin`, `src/ao_elo/dynamic.py:1554`)

İki kontrol vardır. `ln` ve `cap` büyük farkların etkisini sınırlar — 6-0 ile
4-0 arasında fark yoktur. `exp(-|D|/tau)` ise **favori sönümlemesidir**: güçlü
takımın zayıf takımı farklı yenmesi az bilgi taşır, o yüzden az ödüllendirilir.

Field score `90` dakikadır; uzatma oynandıysa `120` dakika. Beraberlikte ve
penaltılarla belirlenen maçta çarpan `1.0`'dir — yani etkisizdir
(`penalty_multiplier = 1.0`, `draw_multiplier = 1.0`).

### 6.3 xG performans düzeltmesi

```text
Q_xG    = tanh((xG_home - xG_away) / xg_scale)          xg_scale = 1.25
Delta_xG = max_xg_ratio * |Delta_base| * Q_xG            max_xg_ratio = 0.30
```

(contract `xg_performance`, `src/ao_elo/xg_live.py:190`)

Uygulanma kuralları — hepsi bağlayıcı:

| Kural | Davranış |
| --- | --- |
| İki takımın xG'si birlikte olmalı | `requires_both_teams = true` |
| Tek taraflı xG | Reddedilir; düzeltme uygulanmaz |
| xG eksik | `FALL_BACK_TO_GOAL_MARGIN_ONLY` — sıfır performans sayılmaz |
| Beraberlik | `draw_behavior = NO_XG_ADJUSTMENT` |
| Penaltı atışları | `penalty_shootout_behavior = NO_XG_ADJUSTMENT`, `shootout_excluded = true` |
| Zaman kapsamı uymuyorsa | `time_scope`: 90 dk, uzatma varsa eşleşen 120 dk xG; uymuyorsa uygulanmaz |
| Winner-gain korkuluğu | `minimum_winner_gain_ratio = 0.70`, analitik sınır |

Winner-gain korkuluğunun anlamı: xG düzeltmesi kazananın kazancını temel
kazancın `%70`'inin altına indiremez. `max_xg_ratio = 0.30` ile bu sınır
analitik olarak zaten sağlanır, bu yüzden contract runtime'da bağlamasını
beklemez (`runtime_floor_expected_to_bind = false`).

---

## 7. Draw ve shootout semantiği

### 7.1 Format-aware draw

```text
raw_draw = draw_at_even * (4 * E_home * (1 - E_home)) ^ draw_shape
limit    = 2 * min(E_home, 1 - E_home)
P_draw   = min(raw_draw, limit)

P_home = E_home - 0.5 * P_draw
P_away = 1 - E_home - 0.5 * P_draw
```

(`src/ao_elo/draw_probability.py:25`, `src/ao_elo/draw_probability.py:53`)

Aktif değerler: `draw_at_even = 0.24`, `draw_shape = 1.00`. Tek maçlık eleme
tie'ında `single_match_draw_at_even = 0.12` kullanılır.

İki özdeşlik **her zaman** korunur:

```text
P_home + P_draw + P_away = 1
P_home + 0.5 * P_draw    = E_home
```

İkincisi kritiktir: draw olasılığını değiştirmek Elo beklenen puanını
değiştirmez, yalnız aynı beklentiyi H/D/A arasında farklı dağıtır. Bu yüzden
tek maç draw'ını `0.12`'ye indirmek rating state'ini değiştirmez
(`one_x_two_probability.rating_state_changed = false`).

`limit = 2 * min(E, 1-E)` terimi, dengesiz maçlarda ham eğrinin negatif
olasılık üretmesini engeller.

**Tek maçlık tie tanımı:** "knockout tie scheduled as exactly one field-score
match". Bu bilgi metadata'dan gelmelidir
(`single_match_metadata_required = true`) — tahmin edilmez.

### 7.2 Shootout

Penaltı atışlarındaki goller **field score'a eklenmez**. Bir tie penaltılarla
belirlendiyse, 90/120 dakikadaki skor neyse odur; penaltı sonucu skoru
değiştirmez.

Bunun tersi de geçerlidir: shootout, eşit olmayan bir 90/120 skorunu
beraberliğe **çevirmez**. İlk maçı 2-1 kazanıp turu penaltılarla kaybeden takım
için o maç hâlâ 2-1 galibiyettir.

Contract bu ayrımı `single_match_advancer = MUST_MATCH_UNEQUAL_FIELD_SCORE_WINNER_UNLESS_SHOOTOUT`
ile pinler: tek maçlık bir tie'da turu geçen takım, skor eşit değilse mutlaka
skorun galibi olmalıdır — istisna yalnız shootout'tur.

---

## 8. AO Live Elo

```text
AO Live Elo = Power Elo + Achievement Reserve + European Progression Bonus
```

(contract `progression_bonus.formula`)

**Achievement Reserve aktif değildir** (`achievement_reserve.active = false`);
katkısı sabit `0`'dır. Formülde durur çünkü kapatılabilir bir katmandır, ama
bugün hiçbir kulübün ratingini etkilemez.

### Progression Bonus

| Kupa | Aşama başına | Sezon tavanı |
| --- | ---: | ---: |
| UCL | `12.0` | `48.0` |
| UEL | `8.0` | `32.0` |
| UECL | `4.0` | `16.0` |

Uygun aşamalar: `ROUND_OF_16`, `QUARTERFINAL`, `SEMIFINAL`, `FINAL` — dört
aşama. Contract `base_bonus = 12.0` ve kupa oranları `1.0 / 0.667 / 0.333`
üzerinden bu değerleri türetir.

Bileşenin altı özelliği:

- **Yalnız R16 ve sonrası.** Eleme ve lig aşaması bonus üretmez.
- **Winner-only** (`winner_only = true`). Yalnız turu geçen alır.
- **Tie başına bir kez** (`single_application_per_tie_id = true`). İki maçlık
  tie'da bonus iki kez verilmez.
- **Kaybedenden düşülmez** (`loser_deduction = false`).
- **Sezon sonunda sıfırlanır** (`season_reset = true`).
- **Sıfır toplamlı değildir.** Yukarıdaki iki maddenin doğrudan sonucu: bonus
  sisteme yeni rating kütlesi ekler. Power Elo'nun sıfır toplamlılığı bozulmaz
  çünkü bonus Power Elo'nun *dışında*, onun üstünde tutulur.

---

## 9. Current AO 1X2

Current AO 1X2, AO rating çekirdeğinden doğrudan üretilen olasılıktır:

```text
E_home = 1 / (1 + 10 ^ ( -(AO_Live_home - AO_Live_away + H) / S ))
(P_home, P_draw, P_away) = score_preserving_1x2(E_home, draw_at_even, draw_shape)
```

(`src/ao_elo/dynamic.py:774`)

**Bu servis edilen nihai olasılık değildir.** Current AO 1X2 üç şeydir birden:
prediction katmanının bir girdisi, karşılaştırma tabanı
(`monitoring.compare_against_current_ao = true`), ve bileşenler çalışmadığında
kullanılan fallback (`fallback = CURRENT_AO_1X2`). Servis edilen olasılık
§12'dedir.

---

## 10. Structural ML

**Aile:** `STRUCTURAL_LOGISTIC` — çok sınıflı lojistik regresyon.

**Feature schema:** `ml_features.FEATURE_SCHEMAS["STRUCTURAL_LOGISTIC"]`,
`33` sayısal ve `4` kategorik alan. Kategorikler: `competition`, `stage`,
`round`, `format_type`.

**Feature sırası doğrulaması:** artifact paketi `feature_schema` ve
`feature_schema_sha256` alanlarını taşır; runtime yüklerken eğitilmiş
`ColumnTransformer`'ın kolon sırasıyla ilan edilen şemayı karşılaştırır. Sıra
veya isim uyuşmazlığı yüklemeyi durdurur.

**Kategorik vokabüler kuralı:** encoder `handle_unknown="ignore"` ile
kurulmuştur. Eğitimde görülmemiş bir kategori **hata vermez**, one-hot'u
sessizce sıfırlanır. Bu yüzden servis edilen her kategorik değerin eğitim
vokabülerinde bulunması testle güvence altına alınır, veriyle değil.
Örnek: UEFA'nın `GROUP` modlu lig aşaması eğitim verisinde `League Stage`
olarak geçer; `League Phase` üretmek modelin hiç görmediği bir kategori yaratır.

**Rolling-window ve exact-UTC:** feature'lar yalnız kickoff'tan önce bilinen
bilgiden türetilir. Aynı exact-UTC kickoff'taki fikstürler tek batch'te
üretilir, böylece batch içi sızıntı olmaz.

### ML iç blend'i

```text
Current ML = AO ^ 0.10 * StructuralLogistic ^ 0.90   (normalize edilir)
```

Kaynak koddan doğrulanmıştır: birleştirme **log-olasılık uzayında** yapılır,
olasılık uzayında değil.

```python
logits  = (1 - weight) * log(max(ao, eps))
logits +=      weight  * log(max(ml, eps))
result  = exp(logits - logits.max(axis=1))
result /= result.sum(axis=1)
```

(`src/ao_elo/ml_prediction.py:122`)

`raw_ml_within_component_weight = 0.9`, contract `current_ml_component`:
`ao_weight = 0.1`, `ml_weight = 0.9`.

---

## 11. AO Domestic Poisson

### Domestic state

Her kulüp için kendi yerel liginden `attack` ve `defence` parametreleri
öğrenilir. State bir checkpoint dosyasında tutulur
(`artifacts/production_prediction/domestic_poisson_state_2026_27.json`,
schema `2.0`) ve `processed_event_ids` listesiyle hangi maçların işlendiğini
kaydeder. Aynı event ID'nin yeniden uygulanması reddedilir
(`domestic_event_id_policy = PERSIST_IDS_REJECT_REAPPLICATION`).

Kulüp parametreleri lig içinde z-skorlanır ve güvenilirlikle çarpılır:

```text
reliability = effective_matches / (effective_matches + shrinkage_matches)
attack      = z(attack_raw) * reliability
```

`shrinkage_matches = 10.0`. Az maçı olan kulüp sıfıra doğru çekilir.

### Transfer katmanı

Yerel parametreler doğrudan Avrupa maçına uygulanmaz; bir transfer modeli
onları Avrupa gol beklentisine çevirir:

```text
mu                  = 0.28040079791632944
elo_slope           = 0.6663036964859297
attack_coefficient  = 0.09209745423072817
defence_coefficient = 0.06065648173017676
venue_coefficient   = 0.0
l2_strength         = 10.0
rho                 = 0.0
```

(contract `prediction_layer.ao_domestic_poisson_component.transfer_config`)

### rho = 0

`rho`, Dixon-Coles düşük skor düzeltmesidir. Production'da `rho = 0`, yani
düzeltme kapalıdır ve skor matrisi bağımsız Poisson'dur. Rho kalibrasyonu
araştırma tarafında mevcuttur ama production'a taşınmamıştır.

### Skor matrisinden 1X2

İki gol beklentisinden bir skor matrisi kurulur, ve hücreler üç sonuca
toplanır: `home > away` toplamı `P_home`, `home == away` köşegeni `P_draw`,
kalanı `P_away`. Bu çekirdek `src/ao_elo/scoreline.py`'dedir ve production
`domestic_poisson` tarafından import edilir — yani `scoreline.py`'nin araştırma
tarafında da kullanılması onu production dışı yapmaz.

### Coverage gate

Her fikstür için iki kulübün domestic profili var mı diye bakılır:

| `domestic_poisson_coverage` | Anlam |
| --- | --- |
| `BOTH` | İki kulübün de profili var |
| `ONE` | Yalnız birinin profili var; mevcut tarafın profili kullanılır, eksik taraf nötrdür |
| `NONE` | Hiçbirinin profili yok; Poisson bileşeni AO tabanına düşer |

(`src/ao_elo/domestic_poisson.py:643`)

Yeterli yerel kanıtı olmayan kulüpler (iki sezon / 40 maç eşiği) takım-bazlı
profilden **çıkarılır** ve `excluded_ao_club_ids` listesine yazılır. `ONE`
kapsamında mevcut tarafın profili kullanılmaya devam eder ve eksik taraf nötr
girdi alır. Yalnız `NONE` kapsamında Poisson bileşeni AO tabanına düşer
(`fallback_to_ao_without_history=True`). 30 Ağustos 2026 üretiminde dağılım
`BOTH=338`, `ONE=54`, `NONE=4`; gerçek satırlarda `ONE` için ölçülen en büyük
`|Poisson - AO|` farkı `0.164`, `NONE` için `0.000` idi.

### Poisson iç blend'i

```text
AO Domestic Poisson = AO ^ 0.50 * Poisson ^ 0.50   (log uzayinda, normalize)
```

contract `ao_domestic_poisson_component`: `ao_weight = 0.5`,
`poisson_weight = 0.5`. Kaynak: `AO_POISSON_RHO0_CONTROL`.

---

## 12. Servis edilen production 1X2

### Nihai formül

```text
Served 1X2 = CurrentML ^ 0.50 * AODomesticPoisson ^ 0.50
```

contract `top_level_blend`: `space = LOG_PROBABILITY`,
`current_ml_weight = 0.5`, `ao_domestic_poisson_weight = 0.5`.
(`src/ao_elo/production_prediction.py:303`)

### Efektif katkılar

İç blend'ler açıldığında:

```text
Served = (AO^0.10 * ML^0.90)^0.50 * (AO^0.50 * P^0.50)^0.50
       =  AO^0.05 * ML^0.45 * AO^0.25 * P^0.25
       =  AO^0.30 * ML^0.45 * P^0.25
```

| Bileşen | Efektif ağırlık |
| --- | ---: |
| Current AO | `0.30` |
| Structural ML | `0.45` |
| Domestic Poisson | `0.25` |

Üstel toplamı `1.00`'dir. Birleştirme log-olasılık uzayında yapılır ve sonunda
olasılıklar toplamı `1` olacak şekilde normalize edilir
(`src/ao_elo/ml_prediction.py:126`).

Log uzayında birleştirmenin anlamı geometrik ortalamadır: bir bileşen bir
sonuca çok düşük olasılık verirse, o sonuç aritmetik ortalamaya göre daha güçlü
bastırılır.

### Fallback davranışı

| Durum | Sonuç |
| --- | --- |
| ML veya Poisson çalışmazsa | `final = Current AO 1X2` |
| `prediction_status` | `ACTIVE_ENSEMBLE` veya `FALLBACK_CURRENT_AO` |
| `fallback_reason` | Fallback satırlarında doldurulur, audit edilir |
| Rating state | Değişmez — `rating_feedback = false` |

(`src/ao_elo/production_prediction.py:326`)

Fallback satırında ML ve Poisson kolonları `NaN` yazılır, sıfır değil — "model
çalıştı ve sıfır dedi" ile "model çalışmadı" birbirinden ayrılır.

---

## 13. Eksik veri ve monitoring

İki sayaç vardır ve **birbirinin yerine geçmezler**:

| Sayaç | Ne ölçer |
| --- | --- |
| `fallback_rate` | `FALLBACK_CURRENT_AO` durumundaki satırların oranı |
| `rows_with_imputed_model_input` | ML çalıştı ama en az bir girdisi impute edildi |

Kritik nokta: **impute edilmiş girdiyle çalışan bir satır `ACTIVE_ENSEMBLE`
döner.** Yani `fallback_rate = 0` olması "bütün tahminler tam veriyle üretildi"
demek değildir. Bir satırın girdisi eksik olabilir, impute edilmiş olabilir ve
yine de tam ensemble olarak servis edilebilir.

Bu yüzden iki sayaç ayrı izlenmelidir. Yalnız `fallback_rate` izlemek, veri
kalitesindeki bozulmayı görünmez kılar.

---

## 14. Aktif ve aktif olmayan katmanlar

Durum tanımları (`docs/ai/RESEARCH_STATUS.md:28`):
`ACTIVE` = contract ve aktif config tarafından kullanılıyor.
`KEEP_SHADOW` = izlenebilir ama canlı ratingi değiştirmiyor.
`REJECTED` = testte zarar veya yapısal sorun.
`INACTIVE` = kodda var, contract'ta kapalı.

| Katman | Durum | Rating'e etkisi | Prediction'a etkisi | Kanıt |
| --- | --- | --- | --- | --- |
| Domestic Surprise | `ACTIVE` | AO First'ü `+/-30` içinde kaydırır | Dolaylı (AO üzerinden) | contract `domestic_surprise.active` |
| Goal margin | `ACTIVE` | Power Elo deltasını çarpar | Dolaylı | contract `goal_margin.active` |
| xG | `ACTIVE` | Deltaya toplanır | Dolaylı | contract `xg_performance.active` |
| Progression bonus | `ACTIVE` | AO Live'a eklenir | Dolaylı | contract `progression_bonus.active` |
| Kupa katkısı | `ACTIVE` | AO First achievement | Dolaylı | contract `domestic_cup_contribution.active` |
| Participation normalization | `ACTIVE` | AO First European prior girdisi | Dolaylı | contract `european_participation.active` |
| Achievement Reserve | `INACTIVE` | Yok (sabit `0`) | Yok | contract `achievement_reserve.active = false` |
| Competition K | `INACTIVE` | Yok; UCL/UEL/UECL çarpanı `1.0` | Yok | contract `competition_k.active = false` |
| Dynamic K | `REJECTED` | Yok | Yok | `RESEARCH_STATUS.md` |
| Season power carry | `INACTIVE` | Yok (`active_power_carry = 0.0`) | Yok | contract `active_power_carry` |
| Team-specific home context | `KEEP_SHADOW` | Yok | Yok | `RESEARCH_STATUS.md` |
| Structural ML | `ACTIVE` | **Yok** | `0.45` efektif ağırlık | contract `current_ml_component` |
| Domestic Poisson | `ACTIVE` | **Yok** | `0.25` efektif ağırlık | contract `ao_domestic_poisson_component` |
| Final prediction ensemble | `ACTIVE_WITH_MONITORING` | **Yok** | Servis edilen 1X2'nin kendisi | contract `prediction_layer` |

Son üç satırdaki "Rating'e etkisi: Yok" tesadüf değildir; sözleşmenin
zorunlu tuttuğu bir invariant'tır (`rating_feedback = false`).

---

## 15. Artifact ve kanıt zinciri

```text
contract
  -> artifact_manifest.path + sha256
       -> Structural Logistic joblib (sha256)
       -> historical Domestic Poisson state (sha256)
       -> live Domestic Poisson state (sha256)
            -> historical_source_data_sha256
            -> live_source_data_sha256
            -> identity_bridge_sha256
            -> coverage_audit_sha256
  -> strict runtime loader
  -> prediction log (contract_sha256 + model fingerprint tasir)
```

Contract manifest'in hash'ini taşır; manifest her artifact'ın hash'ini taşır;
state dosyaları kendilerini üreten girdi CSV'lerinin hash'lerini taşır. Strict
runtime yüklerken zinciri baştan sona doğrular.

**Yakaladığı bozulmalar:** artifact dosyasının değişmesi, contract ile
manifest'in ayrışması, state'in farklı bir girdiden üretilmiş olması,
manifest'te ilan edilenden başka bir modelin yüklenmesi.

**Kanıtlamadığı şeyler:** girdi verisinin *doğru* olduğunu kanıtlamaz — yalnız
ilan edilenle aynı olduğunu kanıtlar. Sağlayıcı yanlış bir skor döndürdüyse
hash zinciri o yanlış skoru sadakatle mühürler. Ayrıca tahminin gerçekten
kickoff'tan önce üretildiğini kanıtlamaz; bunun için append-only ledger gerekir
(§17).

---

## 16. Güncel değerlendirme sonuçları

Kaynak: `reports/current_model/` ve `reports/production_prediction/` paketleri.

### Veri penceresi

```text
gelistirme penceresi   2018/19 - 2025/26
outer test foldlari    2020/21 - 2025/26  (alti fold)
unseen mac             4,884
```

### Rating çekirdeği

| Model | Brier | Log-loss | Accuracy | Spearman | Pairwise |
| --- | ---: | ---: | ---: | ---: | ---: |
| `REFERENCE_CORE_NO_ACTIVE_EXTRAS` | `0.568053` | `0.959174` | `0.554464` | `0.681487` | `0.758195` |
| `CURRENT_PRODUCTION` | `0.566413` | `0.956259` | `0.559173` | `0.683258` | `0.759421` |

### Prediction katmanı

| Model | Brier | Log-loss | Accuracy |
| --- | ---: | ---: | ---: |
| `CURRENT_AO` | `0.566413` | `0.956259` | `0.559173` |
| `CURRENT_ML_BLEND` | `0.563746` | `0.952686` | `0.558559` |
| `AO_POISSON_BLEND` | `0.564488` | `0.953443` | `0.560606` |
| `AO_POISSON_RHO0_CONTROL` | `0.564313` | `0.953034` | `0.561016` |
| `ML_POISSON_ENSEMBLE` | `0.563050` | `0.951368` | `0.559582` |

Ensemble'ın Current AO'ya farkı: Brier `-0.003362`, log-loss `-0.004891`,
accuracy `+0.000410`.

### Güven aralıkları

Conservative envelope (`4,000` bootstrap), `ML_POISSON_ENSEMBLE` vs `CURRENT_AO`:

```text
Brier     -0.003362   %95 CI [-0.005559, -0.001353]   guvenilir iyilesme
Log-loss  -0.004891   %95 CI [-0.008308, -0.001447]   guvenilir iyilesme
```

`CURRENT_ML_BLEND` tabanına karşı ise aralık sıfırı keser
(`tie_or_match` yönteminde `[-0.001630, +0.000276]`), yani ensemble'ın kendi ML
koluna üstünlüğü güvenilir değildir.

### Kararlar

| Karar | Değer |
| --- | --- |
| Otomatik gate (`selected_candidate.json`) | `KEEP_SHADOW` |
| `production_activated` | `false` |
| `uncertainty_gate` | `false` |
| Contract'taki ürün kararı | `PROMOTE_WITH_MONITORING` |

Bu ikisi çelişki değildir ama karıştırılmamalıdır: otomatik kapı ensemble'ı
belirsizlik testinde geçirmediği için `KEEP_SHADOW` işaretler; production'da
aktif olması **manuel bir ürün kararıdır**, otomatik terfi değildir.

### External benchmark

`363` eşleşmiş maçta, ClubElo'ya karşı:

| Kol | Brier | Log-loss | Accuracy |
| --- | ---: | ---: | ---: |
| `CLIMATOLOGY_WALK_FORWARD` | `0.642983` | `1.062634` | `0.457300` |
| `CLUBELO_PUBLISHED_SCALE_400` | `0.574983` | `0.966819` | `0.537190` |
| `AO_RATING_CORE_1X2` | `0.577244` | `0.972194` | `0.553719` |
| `AO_SERVED_ENSEMBLE_50_50` | `0.580864` | `0.975477` | `0.556474` |

Bu küçük örneklemde AO, ClubElo'nun **gerisindedir** (Brier farkı `+0.0059`).
Accuracy'de öndedir. `363` maç bu farkı ayırt etmek için yeterli değildir ve bu
sonuç tam pencere veya prospective kanıt yerine geçmez.

### Contract'taki donmuş kanıt ile karıştırılmaması gereken

Contract'ın `static_initial_elo` bloğu `brier = 0.569287`,
`log_loss = 0.960299`, `accuracy = 0.554259` taşır. Bunlar **exposure-cap
kararının kendi ölçümüdür** (`2026-08-20`), yukarıdaki güncel backtest
metrikleri değildir; contract bunu `evidence_note` ile açıkça işaretler. İki
sayı kümesi birbirinin yerine kullanılmamalıdır.

---

## 17. Kanıt sınırları

- **Retrospective backtest prospective kanıt değildir.** Yukarıdaki bütün
  metrikler geçmiş veriye geriye dönük uygulanmış walk-forward ölçümlerdir.
- **Nested/OOS sonuç sabit production blend'inin birebir replay'i olmayabilir.**
  Değerlendirmedeki ensemble kolu fold bazında seçim yapar; production sabit
  `%50/%50` kullanır. İkisi aynı sayı değildir.
- **CI'nin sıfırı kesmesi ürün kararını otomatik geçersiz kılmaz.** Ensemble'ın
  kendi ML koluna üstünlüğü güvenilir değildir; bu, katmanın kapatılması
  gerektiği anlamına gelmez, ama "ölçülmüş kazanç" diye sunulamayacağı anlamına
  gelir.
- **Append-only pre-match ledger olmadan untouched holdout iddiası kurulamaz.**
  Mevcut prediction CSV'si yeniden üretilebilir bir çıktıdır; tahminin
  kickoff'tan önce yazıldığını kanıtlamaz.
- **Provider coverage gate eksiksizlik kanıtı değildir.** Beklenen maç sayısı
  moddan veya medyandan çıkarılan bir tahmindir
  (`RECURRING_MODE_ELSE_CEIL_MEDIAN_INFERRED_NOT_VERIFIED`), resmî fikstür
  kaydı değildir. Kapının verdiği karar hangi kaynağın seçildiğine bağlıdır.
- **Repo tek başına canlı ingestion servisi değildir.** §19.

---

## 18. Holdout protokolü — 2026-09-08 sonrası donmuş değerler

`docs/HOLDOUT_PROTOCOL_2026_27.md` §5, kilit sonrasında aşağıdakilerin
değiştirilmesini yasaklar.

### AO First seed

| Parametre | Donmuş değer |
| --- | ---: |
| `max_european_exposure` | `0.65` |
| `minimum_domestic_prior_weight` | `0.35` |
| `european_participation_shrinkage` (k) | `0.20` |
| `cup_contribution_weight` (w) | `0.12903225806451613` |
| `unknown_league_finish_score` | `0.15` |
| Domestic Surprise theta / penalty / cap | `0.40` / `0.50` / `+/-30` |

### Rating çekirdeği

| Parametre | Donmuş değer |
| --- | ---: |
| `elo_scale` | `835.5614973262034` |
| `home_advantage` | `148.54426619132505` |
| `k_factor` | `103.98098633392752` |
| `active_power_carry` | `0.0` |
| Qualifier base importance | `0.40 / 0.55 / 0.70 / 0.85` |
| Qualifier retention | `0.50` |
| Effective qualifier K | `0.20 / 0.275 / 0.35 / 0.425` |
| MAIN multiplier | `1.00` |
| Goal alpha / tau / cap | `0.15` / `300` / `4` |
| xG ratio / scale / min winner gain | `0.30` / `1.25` / `0.70` |
| Progression artışları / tavanları | `12/8/4` / `48/32/16` |

### 1X2 ve blend

| Parametre | Donmuş değer |
| --- | ---: |
| `draw_at_even` | `0.24` |
| `single_match_draw_at_even` | `0.12` |
| `draw_shape` | `1.00` |
| Top-level blend | `0.50` ML / `0.50` Poisson |
| ML iç blend | `0.10` AO / `0.90` Structural |
| Poisson iç blend | `0.50` AO / `0.50` Poisson |
| `rho` | `0.0` |
| Efektif AO / ML / Poisson | `0.30` / `0.45` / `0.25` |

Ayrıca yasak: kapalı katmanları açmak, feature schema'yı sonuçtan yararlanacak
biçimde genişletmek, eski prediction logunu yeni modelle yeniden üretip
prospective diye sunmak.

Acil bir yazılım hatası düzeltilmek zorundaysa yeni bir production revision,
artifact fingerprint ve etkilenen maç aralığı açıkça kaydedilir; eski ve yeni
revision sonuçları tek homojen holdout gibi birleştirilmez.

---

## 19. Canlı işletim akışı

### Sıra

```text
1. fixture ingestion       fikstur listesi ve kickoff'lar
2. identity cozumleme      provider ID -> permanent club_id
3. state checkpoint        domestic Poisson state guncellemesi
4. pre-match feature build kickoff oncesi ozellikler
5. prediction lock         kickoff - 10/15 dk, append-only ledger
6. audit log               status, fallback_reason, fingerprint
7. monitoring              fallback_rate + imputed-input sayaci
8. sonuc replay            mac bitince rating guncellemesi
```

### Repo içinde mevcut olanlar

- Feature build ve prediction üretimi (`scripts/build_2026_27_prediction_features.py`)
- Domestic state checkpoint üretimi ve replay
- Artifact build ve hash zinciri doğrulaması
- Coverage audit ve identity bridge
- Prediction log şeması ve audit alanları

### Harici servis tarafından sağlanması gerekenler

Aşağıdakiler bu repository'de **yoktur**:

- Fetch worker ve provider rate-limit/retry yönetimi
- Kalıcı veritabanı
- Identity servisi (birleşme, taşınma, yeni hukuki kulüp için manuel karar
  gerekir — `LIVE_DATA_INGESTION.md:90`)
- Prediction lock kuyruğu ve append-only ledger
- Alarm ve nöbet mekanizması
- Read API

Fixture polling için önerilen varsayılan (`LIVE_DATA_INGESTION.md:405`):
`7+` gün uzakta günde bir, `1-7` gün arası altı saatte bir, `0-24` saat saatte
bir, son `60` dakikada `T-60` ve `T-15` doğrulaması. Lock sonrası fikstür
değişirse iptal/versiyon event'i gerekir.

### İzlenmesi gereken sayaçlar

| Sayaç | Neden |
| --- | --- |
| `fallback_rate` | Bileşen arızasını gösterir |
| `rows_with_imputed_model_input` | Veri kalitesi bozulmasını gösterir; fallback'e girmez |
| `domestic_poisson_coverage` dağılımı | `NONE`/`ONE` oranındaki artış kimlik veya kapsam sorunudur |
| Servis edilen vs Current AO farkı | Ensemble'ın katkısının izlenmesi (`compare_against_current_ao`) |
| Artifact hash uyuşmazlığı | Zincir bozulması; strict modda yükleme durur |

---

## 20. Kısa sözlük

| Terim | Tanım |
| --- | --- |
| **AO First Elo** | Sezon başı kulüp gücü. Domestic Prior ile European Prior'ın exposure ağırlıklı karışımı. |
| **Power Elo** | Maçtan maça güncellenen, sıfır toplamlı rating çekirdeği. |
| **Achievement Reserve** | AO Live formülünde yer alan, bugün kapalı ve sabit `0` olan bileşen. |
| **Progression Bonus** | R16 ve sonrasında turu geçene verilen, sıfır toplamlı olmayan, sezon sonunda sıfırlanan ek. |
| **AO Live Elo** | `Power Elo + Achievement Reserve + Progression Bonus`. Servis edilen kulüp gücü. |
| **Current AO 1X2** | AO Live Elo'dan doğrudan türetilen H/D/A. Karşılaştırma tabanı ve fallback. |
| **Current ML** | `AO^0.10 * StructuralLogistic^0.90`, log uzayında. |
| **AO Domestic Poisson** | `AO^0.50 * Poisson^0.50`, log uzayında. |
| **Served 1X2** | `CurrentML^0.50 * AODomesticPoisson^0.50`. Efektif olarak `AO^0.30 * ML^0.45 * P^0.25`. |

---

## Düzeltilmesi gereken dokümantasyon çelişkileri

Bu belgeyi yazarken bulunan, üst otorite uygulanarak çözülen noktalar:

1. **`static_initial_elo.rating_feedback = true` vs
   `prediction_layer.rating_feedback = false`.** İkisi farklı şeyi kastediyor:
   birincisi exposure-cap parametresinin ratingi etkilediğini, ikincisi
   prediction çıktısının ratinge geri beslenmediğini söylüyor. Aynı anahtar adı
   iki farklı anlamda kullanılıyor ve okuyucuyu yanıltmaya açık. Alan adlarından
   biri netleştirilmelidir.

2. **`selected_candidate.json` `decision = KEEP_SHADOW` vs contract
   `PROMOTE_WITH_MONITORING`.** Çelişki değil — biri otomatik kapı, diğeri ürün
   kararı. Ama iki dosyanın hiçbirinde bu ayrımın açıklaması yok; yalnız
   `RESEARCH_STATUS.md` bahsediyor. Contract'a bir `decision_kind` notu
   eklenmesi bu karışıklığı kapatır.

3. **`normalize_stage` üç ad birden kabul ediyor:** `Group Stage`,
   `League Stage`, `League Phase` (`src/ao_elo/dynamic.py:1580`). Ama ML feature
   vokabüleri yalnız ilk ikisini tanır; `League Phase` üretmek kategorik
   özelliği sessizce sıfırlar. `dynamic.py`'nin toleransı ile ML vokabülerinin
   katılığı arasındaki bu fark belgede yazılı değildi — §10'a eklendi, ancak
   `dynamic.py`'nin `League Phase`'i kabul etmeye devam etmesi ileride aynı
   hatayı tekrar mümkün kılar.

4. **`achievement_cap = 1.10` erişilebilir maksimum sanılmaya açık.** Aktif
   parametrelerle ulaşılabilir en yüksek achievement `1.0800`'dir, yani cap hiç
   bağlamaz. Contract ve README'de "cap" olarak geçmesi bunu bir hedef değer
   gibi gösteriyor; §4.8'de korkuluk olduğu açıkça yazıldı.

5. **External benchmark'ta AO, ClubElo'nun gerisinde** (`0.580864` vs
   `0.574983` Brier). Bu sonuç README §6'da yer almıyor. `363` maçlık örneklem
   küçük olduğu için sonuç belirleyici değil, ama tek yönlü sunum riski var;
   §16'da her iki yön de yazıldı.
