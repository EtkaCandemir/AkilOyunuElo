# Aktif Model Formulleri

Bu belge production modelinin matematiksel referansidir. Tum degerler aktif
contract ve kaynak koddan alinmistir.

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
| `K` | Match update buyuklugu, `103.9809863339` |

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
LeagueFinish = 0.10, aksi halde
```

Kupa ve duble:

```text
CupBase = 0.62, is_cup_winner=true ise; aksi halde 0

DoubleBonus = 0.08 * LeagueFinish,
              yalniz champion AND cup_winner ise

AchievementUncapped = max(LeagueFinish, CupBase) + DoubleBonus
DomesticAchievement = min(1.10, AchievementUncapped)
```

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

u_europe = ln(1 + WeightedEuropeanHistory) / ln(1 + 20)
```

Aktif `european_tail_beta=0`:

```text
EuropeanHistoryNorm = min(u_europe, 1)

EuropeanPrior = 500
              + 1559.714795008913 * EuropeanHistoryNorm
```

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
e_eff = e                                      , e <= 0.85
        0.85 + beta_exp * (e - 0.85)           , e > 0.85
```

Aktif `beta_exp=0`:

```text
e_eff = min(e, 0.85)
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

## 13. Klasik Result Residual

```text
S_home = 1   home field-score win
S_home = .5  field-score draw
S_home = 0   away field-score win

r_base = S_home - E_home
Delta_base = K * r_base
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
Delta_GD_bonus = K * r_base * (M_GD - 1)
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
Delta_xG = K * r_xG
```

Birlesik Power Delta:

```text
r_final = r_base * M_GD + r_xG

Delta_final = K * r_final
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
