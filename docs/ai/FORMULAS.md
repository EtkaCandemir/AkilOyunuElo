# Aktif Model Formulleri

Bu belge production modelinin matematiksel referansidir. Tum degerler aktif
contract ve kaynak koddan alinmistir.

28 Agustos 2026 bugfix revision'u bu belgedeki sayisal formulleri veya donmus
parametreleri degistirmez. `played_*` icin canonical string boolean once 0/1
olarak okunur; agirlikli exposure toplami aynidir. Gecersiz zaman/skor/metadata
girdilerinin ret ve fallback kurallari `DATA_CONTRACTS.md` icindedir.

## 1. Notasyon

| Sembol | Anlam |
| --- | --- |
| `w_i` | Bes sezon agirligi |
| `B_c` | Country benchmark, `25` |
| `B_e` | European history benchmark, `20` |
| `L` | League Strength |
| `A` | Domestic Achievement Score |
| `e` | Ham European Exposure |
| `e_eff` | Effective European Exposure |
| `R_h, R_a` | Pre-match AO Live Elo |
| `D` | Home-away farki ve saha avantaji dahil effective rating farki |
| `E` | Ev sahibinin beklenen mac puani |
| `S` | Ev sahibinin gercek mac puani, `1/0.5/0` |
| `K` | Match update buyuklugu, `103.98098633392752` |

Bes sezon agirliklari:

```text
w = (0.07, 0.13, 0.20, 0.27, 0.33)
sum(w) = 1
```

## 2. V2 Gorunur Olcek

V1.1'den v2 referans olcegine affine multiplier:

```text
M = 1500 / (903.92 - 500)
  = 3.713606654783126

R_v2 = 500 + M * (R_v1 - 500)
```

Aktif component donusumleri:

```text
Domestic League Component      = 140 * M = 519.9049316696377
Domestic Achievement Component = 160 * M = 594.1770647653002
European Prior Max Boost       = 420 * M = 1559.714795008913
Scale                          = 225 * M = 835.5614973262034
H                              =  40 * M = 148.54426619132505
K                              =  28 * M = 103.98098633392752
```

`500-2000` hard sinir degil referans bandidir. Static veya dynamic rating
sonunda clipping yoktur. Affine Scale/H/K donusumu, yalniz gorunur puan
araligini buyuturken temel Elo olasilik geometrisini korur.

## 3. Country Strength ve League Strength

```text
WeightedCountry = sum_i(w_i * country_points_i)

u_country = ln(1 + WeightedCountry) / ln(1 + B_c)
```

Genel upper-tail fonksiyonu:

```text
Tail(u, beta) = u                         , u <= 1
                1 + beta * (u - 1)        , u > 1
```

Aktif `country_tail_beta=0` oldugu icin:

```text
country_norm = min(u_country, 1)
LeagueStrength = country_norm ^ 0.80
```

## 4. Domestic Achievement

Bilinen lig pozisyonu `p`, takim sayisi `N>1`:

```text
PositionPercentile = (N - p) / (N - 1)

PercentileScore = 0.15 + 0.70 * PositionPercentile ^ 1.00
ChampionBase    = 1.00, sampiyonsa; aksi halde 0
LeagueFinish    = max(PercentileScore, ChampionBase)
```

Pozisyon bilinmiyorsa:

```text
LeagueFinish = 1.00, is_league_champion=true ise
LeagueFinish = 0.15, aksi halde
```

Bilinmeyen pozisyon percentile egrisinin **tabaninda** durur. Daha dusuk bir
deger, kanit yoklugunu mumkun olan en kotu kanittan (bilinen son siradan, ki o
da `0.15`tir) daha agir cezalandirirdi. Onceki `0.10` bu monotonlugu bozuyordu.

Kupa katkisi:

```text
CupBase = 0.62, is_cup_winner=true ise; aksi halde 0

CupContribution = w * min(LeagueFinish, CupBase),   aktif w = 0.129032258065

AchievementUncapped = max(LeagueFinish, CupBase) + CupContribution
DomesticAchievement = min(1.10, AchievementUncapped)
```

`max` tek basina kupayi bir **taban** yapar: lig skoru kupa tabaninin uzerinde
olan kupa sahibi kupasindan sifir kredi alir. Katki terimi bu boslugu kapatir
ve iki basarinin zayif olanini gercek bir katki olarak ekler.

Aktif agirlik turetilmistir:

```text
w = cup_double_bonus_multiplier * champion_base_score / cup_base_score
  = 0.08 * 1.00 / 0.62
  = 0.129032258065
```

Bu secim, onceki kuralin **zaten odullendirdigi** grubu birebir korur:
sampiyon-ve-kupa toplami `1.00 + 0.08 = 1.0800`, yeni kuralda
`1.00 + 0.129032 * 0.62 = 1.0800`. Degisiklik saf bicimde "ayni mantigi
sampiyon olmayanlara da uygula" demektir.

Invariantlar:

- Kupa kazanmayan takimda `min(L, C) = 0`, dolayisiyla katman **inerttir**.
- Katman achievement'i asla dusurmez.
- `achievement_cap = 1.10` bu agirlikta baglayici degildir; ulasilabilir
  maksimum `1.0800`dir.

Sampiyonluk egride bilincli bir step'tir. Sampiyon olmayan lig birincisi
verisi validation tarafinda tutarsiz sayilmalidir; champion flag bilinen
pozisyonla uyumlu tutulur.

## 5. Domestic Prior

```text
AchievementScale = 0.40 + 0.60 * LeagueStrength

DomesticPrior = 500
              + 519.9049316696377 * LeagueStrength
              + 594.1770647653002
                * DomesticAchievement
                * AchievementScale
```

Achievement etkisi zayif ligde tamamen sifirlanmaz; minimum scale `0.40`tir.
Guclu ligde scale `1.00`a ulasir.

## 6. European History ve European Prior

```text
WeightedEuropeanHistory = sum_i(w_i * club_points_i)
```

### 6.1 Katilim normalizasyonu

Girilmeyen sezon `club_points = 0` katkisi yapar ve bu, girilip hic puan
alinamayan sezonla aritmetik olarak aynidir. Prior bu ikisini ayirt edemedigi
icin agirlikli toplam, kulubun gercekten girebildigi agirliga gore
normalize edilir:

```text
pw = WeightedSeasonExposure = sum_i(w_i * Played_i)

EuropeanHistoryRate = WeightedEuropeanHistory * (1 + k) / (pw + k),  pw > 0
EuropeanHistoryRate = 0,                                             pw = 0

active k = 0.20
```

`(1 + k)` payda degil **pay**dadir; bu yuzden `pw = 1` iken oran yayinlanmis
history'nin birebir aynisidir ve bes sezonluk tam kaniti olan kulup **tanim
geregi hic hareket etmez**. Duzeltme yalniz gercek katilim boslugu kadardir ve
ratingi asla dusuremez.

`pw = 0` olan kulup sifir oran tasir. O satirlarda `european_exposure = 0`
oldugu icin prior blend'e zaten girmez; bos paydada bir oran uretmek gurultu
olurdu.

`k`, kucuk paydada carpanin patlamasini engelleyen emniyet sabitidir:

| `pw` | `k=0` | `k=0.20` | `k=0.75` |
| ---: | ---: | ---: | ---: |
| 0.07 | `14.29x` | `4.44x` | `2.13x` |
| 0.20 | `5.00x` | `3.00x` | `1.84x` |
| 0.53 | `1.89x` | `1.64x` | `1.37x` |
| 1.00 | `1.00x` | `1.00x` | `1.00x` |

Katman kapaliyken `EuropeanHistoryRate = WeightedEuropeanHistory`.

### 6.2 Normalizasyon ve prior

```text
u_europe = ln(1 + EuropeanHistoryRate) / ln(1 + 20)
```

Aktif `european_tail_beta=0`:

```text
EuropeanHistoryNorm = min(u_europe, 1)

EuropeanPrior = 500
              + 1559.714795008913 * EuropeanHistoryNorm
```

Exposure agirligina dokunulmaz: katilim yalniz prior'in **girdisini** duzeltir,
blend agirligini degil.

## 7. European Exposure

Her sezon icin:

```text
Played_i = 1 veya 0
MatchExposure_i = min(1, matches_i / match_cap_i)
```

Agirlikli kanitlar:

```text
SeasonExposure = sum_i(w_i * Played_i)
MatchExposure  = sum_i(w_i * MatchExposure_i)

e = 0.60 * SeasonExposure + 0.40 * MatchExposure
```

Genel tail:

```text
e_eff = e                                      , e <= 0.65
        0.65 + beta_exp * (e - 0.65)           , e > 0.65
```

Aktif `beta_exp=0`:

```text
e_eff = min(e, 0.65)
```

Rating source etiketi ham `e` ile belirlenir:

```text
e = 0       -> Pure Domestic Projection
0 < e < .75 -> Mixed Domestic-European Estimate
e >= .75    -> European Evidence-Based Rating
```

## 8. AO First Elo Blend

Domestic Surprise oncesi:

```text
AOFirst_pre = DomesticPrior
            + e_eff * (EuropeanPrior - DomesticPrior)
```

Esdeger agirlik yazimi:

```text
AOFirst_pre = (1 - e_eff) * DomesticPrior + e_eff * EuropeanPrior
```

## 9. Variance-Controlled Domestic Surprise

Guncel ve history finish percentile:

```text
P = (N - position) / (N - 1)
```

Bes tam history gozlemi icin:

```text
P_bar = sum_i(w_i * P_i)
Variance = sum_i(w_i * (P_i - P_bar)^2)
Volatility = sqrt(Variance)

NormalizedVolatility = min(1, 2 * Volatility)
Consistency = 1 - 0.50 * NormalizedVolatility

RawSurprise = P_current - P_bar
EffectiveSurprise = RawSurprise * Consistency
```

Domestic adjustment:

```text
RawDomesticAdjustment = 594.1770647653002
                      * AchievementScale
                      * 0.40
                      * EffectiveSurprise

DomesticAdjustment = clip(RawDomesticAdjustment, -30, +30)
AdjustedDomesticPrior = DomesticPrior + DomesticAdjustment
```

AO First etkisi:

```text
AOFirst = AOFirst_pre + (1 - e_eff) * DomesticAdjustment
```

Esdeger ifade:

```text
AOFirst = AdjustedDomesticPrior
        + e_eff * (EuropeanPrior - AdjustedDomesticPrior)
```

Bes sezon eksikse veya katman kapaliysa `DomesticAdjustment=0`.

## 10. AO Live Elo

```text
ProgressionTotal = Bonus_UCL + Bonus_UEL + Bonus_UECL

AOLive = PowerElo + AchievementReserve + ProgressionTotal
```

Aktif `AchievementReserve=0`.

## 11. Beklenen Puan

```text
D = R_home - R_away + H_effective

H_effective = 148.54426619132505, normal saha
H_effective = 0, neutral saha

E_home = 1 / (1 + 10 ^ (-D / 835.5614973262034))
```

`E_home`, galibiyet olasiligi degil, beraberligi yarim puan sayan beklenen
puandir.

## 12. Score-Preserving 1X2

Normal/two-leg mac:

```text
draw_at_even = 0.24
```

Tek maclik knockout tie:

```text
draw_at_even = 0.12
```

Aktif `shape=1.00`:

```text
P_draw_raw = draw_at_even * (4 * E_home * (1 - E_home)) ^ shape
P_draw_cap = 2 * min(E_home, 1 - E_home)
P_draw = min(P_draw_raw, P_draw_cap)

P_home = E_home - 0.5 * P_draw
P_away = 1 - E_home - 0.5 * P_draw
```

Invariantlar:

```text
P_home + P_draw + P_away = 1
P_home + 0.5 * P_draw = E_home
tum olasiliklar >= 0
```

### 12.1 Production ML + Domestic Poisson Tahmini

Uc sinifli iki olasilik vektoru icin log-probability blend:

```text
LogBlend(P, Q, w)_c =
    exp((1-w) * ln(P_c) + w * ln(Q_c))
    / sum_j exp((1-w) * ln(P_j) + w * ln(Q_j))
```

Aktif katman:

```text
P_current_ml = LogBlend(P_AO, P_structural_logistic, 0.90)

P_ao_poisson = LogBlend(
    P_AO,
    P_domestic_poisson_rho_0,
    0.50
)

P_served = LogBlend(P_current_ml, P_ao_poisson, 0.50)
```

Domestic Poisson transfer katsayilari:

```text
mu                  = 0.28040079791632944
elo_slope           = 0.6663036964859297
attack_coefficient  = 0.09209745423072817
defence_coefficient = 0.06065648173017676
venue_coefficient   = 0
l2_strength         = 10
rho                 = 0
```

Bu olasilik katmani `E_home`, Power Delta veya AO Live Elo state'ini
degistirmez. Herhangi bir artifact, model feature'i veya Domestic Poisson state
sorununda `P_served=P_AO` olur ve fallback nedeni audit loguna yazilir.

### 12.2 Domestic veri kabul beklentisi

Domestic veri kabul notu: secondary sezonsal mac sayisi icin tekrar eden mod
varsa o deger, yoksa `ceil(median(counts))` beklenen sayimdir; kabul icin
`count / expected >= 0.95` gerekir. Bu bir coverage tahminidir, rating veya
Poisson parametresi degildir. Kaynak/provider sezonu AO sezonundan turetilmez.

## 13. Klasik Result Residual

Once mac asamasina gore effective K secilir:

```text
K_effective = 103.98098633392752 x BaseStageImportance x QualifierRetention

                         Base    Retention   Effective
Q1                       0.40       0.50       0.200
Q2                       0.55       0.50       0.275
Q3                       0.70       0.50       0.350
Qualifying Play-off      0.85       0.50       0.425
League phase ve sonrasi  1.00       1.00       1.000
```

Preliminary Round `Q1` kullanir. UCL/UEL/UECL arasinda dusus state'i
sifirlamaz. Retention her qualifier macinin K degerine pozitif/negatif simetrik
olarak gomuludur. Ana asamaya geciste Power state degismez; mac veya yeni bir
rating olayi yoksa kullaniciya gosterilen Elo da degisemez.

```text
S_home = 1   home field-score win
S_home = .5  field-score draw
S_home = 0   away field-score win

r_base = S_home - E_home
Delta_base = K_effective * r_base
```

## 14. Kontrollu Gol Farki

Mutlak field-score farki `GD`:

```text
M_GD = 1, draw / shootout / GD<=1 ise

M_GD = 1
     + 0.15 * ln(min(GD,4)) * exp(-abs(D)/300), aksi halde
```

Goal-margin residual:

```text
r_GD = r_base * M_GD
Delta_GD_bonus = K_effective * r_base * (M_GD - 1)
```

`D=0` icin ornek multiplier'lar:

| Gol farki | Multiplier |
| ---: | ---: |
| 1 | `1.0000` |
| 2 | `1.1040` |
| 3 | `1.1648` |
| 4+ | `1.2079` |

`abs(D)` arttikca `exp(-abs(D)/300)` bonusu sondurur.

## 15. Bounded xG Performans Duzeltmesi

Yalniz uygun, iki tarafli xG ve decisive field result icin:

```text
Q_xG = tanh((xG_home - xG_away) / 1.25)

r_xG = 0.30 * abs(r_base) * Q_xG
Delta_xG = K_effective * r_xG
```

Birlesik Power Delta:

```text
r_final = r_base * M_GD + r_xG

Delta_final = K_effective * r_final
            = Delta_base + Delta_GD_bonus + Delta_xG
```

Home galibiyetinde home xG avantaji pozitifse Delta buyur; xG dezavantaji
varsa azalir. Away galibiyetinde isaretler simetrik calisir. `%30` ratio ve
`|tanh|<1` nedeniyle xG tek basina klasik result residual'in yonunu tersine
ceviremez:

```text
winner result gain >= 0.70 * classic result gain
```

Bu nedenle `minimum_winner_gain_ratio=0.70` runtime'da ek guvenlikten cok
analitik sinirin acik ifadesidir.

Draw, shootout veya eksik xG:

```text
r_xG = 0
```

## 16. Power Conservation

```text
Power_home_post = Power_home_pre + Delta_final
Power_away_post = Power_away_pre - Delta_final

sum_power_post = sum_power_pre
```

Tolerans `1e-9`.

## 17. Progression Bonus

Eligible stage set:

```text
ROUND_OF_16, QUARTERFINAL, SEMIFINAL, FINAL
```

Competition ratio:

```text
ratio_UCL  = 1
ratio_UEL  = 2/3
ratio_UECL = 1/3
```

Tie tamamlandiginda:

```text
RequestedBonus = 12 * competition_ratio
AppliedBonus = min(RequestedBonus, CompetitionCap - CurrentCompetitionBonus)
```

| Turnuva | Stage basi | Sezon cap |
| --- | ---: | ---: |
| UCL | `12` | `48` |
| UEL | `8` | `32` |
| UECL | `4` | `16` |

Kaybeden duzeltmesi `0`dir. Bu nedenle progression AO Live toplaminda
non-zero-sum'dir. Ayni `tie_id` yalniz bir kez uygulanir ve yeni sezona tasinmaz.

## 18. Loss Metrikleri

Gercek sinif one-hot `y=(y_H,y_D,y_A)`, tahmin `p=(p_H,p_D,p_A)`:

```text
Brier_1X2 = (p_H-y_H)^2 + (p_D-y_D)^2 + (p_A-y_A)^2

LogLoss_1X2 = -ln(p_actual)

Accuracy = 1[argmax(p) = actual_class]
```

Bu projede raporlanan standard multiclass Brier, uc kare hatanin toplamidir;
uc sinifa bolunmez.
