# AO European Elo Güncel Production Model Değerlendirmesi

## Teknik özet

- Production contract hesap parametreleri final-candidate ile eşittir: `True`. Birebir JSON eşitliği `False`; fark production'a sonradan eklenen açıklayıcı formül alanlarıdır.
- Anlamlı karşılaştırma için aynı Scale/H/K üzerinde bütün aktif ek katmanları kapalı `REFERENCE_CORE_NO_ACTIVE_EXTRAS` kolu üretildi.
- `6` unseen/development fold sezonunda `4884` maç değerlendirildi. Güncel model Brier `0.572740`, log-loss `0.965972`, accuracy `0.5504` üretti.
- Referansa karşı farklar: Brier `-0.000958`, log-loss `-0.001398`, accuracy `+0.0031`.
- Fold kazanımları Brier `6/6`, log-loss `6/6`, aynı-sezon Spearman `6/6`, pairwise `6/6`.
- Production kararı: **KEEP**. Model çalışır ve guardrail'ler geçer; ancak 2026/27 lig aşaması sonrası prospective holdout henüz tamamlanmadığı için yeni bir PROMOTE iddiası yapılmamalıdır.

## Güncel sözleşme ve aktif mimari

- Model: `ao-european-elo-v2.0-dev-freeze`; production revision `2026-08-06-penalty-field-score-goal-alpha-015-bounded-xg`; final candidate `ao-european-elo-v2.0-final-candidate-2026-08-05`.
- Statik: country benchmark `25`, European history benchmark `20`, sezon ağırlıkları `0.07/0.13/0.20/0.27/0.33`, country/european/exposure tail beta `0/0/0`.
- Domestic Prior = 500 + lig gücü bileşeni + lig/kupa başarısının lig gücüyle ölçeklenmiş bileşeni. Şampiyonluk, kupa ve duble kuralları başlangıç ratinginde kullanılır.
- Domestic Surprise aktiftir: theta `0.4`, variance penalty `0.5`, cap `+/-30.0`, tam geçmiş `5` sezon.
- Dynamic: Scale `835.561497`, H `148.544266`, K `103.980986`, carry `0.0`.
- Gol farkı: alpha `0.15`, tau `300.0`, GD cap `4`.
- xG: ratio `0.3`, scale `1.25`, winner floor `0.7`; iki taraf xG yoksa GD-only fallback.
- Progression: UCL/UEL/UECL `12/8/4`, sezon cap `60/40/20`, winner-only, tek tie uygulaması, sezon resetli.
- Achievement Reserve, Competition K, Dynamic K, season carry ve takım bazlı home context aktif değildir. Ev sahibi avantajı global H olarak uygulanır.

## Baseline ve güncel model ana metrikleri

| model | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy | xg_applied_matches | total_progression_bonus | maximum_abs_match_delta | delta_vs_reference_brier_1x2 | delta_vs_reference_log_loss_1x2 | delta_vs_reference_accuracy_1x2 | delta_vs_reference_same_season_spearman | delta_vs_reference_same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CURRENT_PRODUCTION | 4884 | 0.5727403 | 0.9659716 | 0.5503686 | 0.6720790 | 0.7535029 | 600 | 2892.0000000 | 110.9260891 | -0.0009584 | -0.0013978 | 0.0030713 | 0.0040202 | 0.0014788 |
| REFERENCE_CORE_NO_ACTIVE_EXTRAS | 4884 | 0.5736987 | 0.9673694 | 0.5472973 | 0.6680588 | 0.7520241 | 0 | 0.0000000 | 94.8104820 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| ABLATION_NO_DOMESTIC_SURPRISE | 4884 | 0.5732295 | 0.9666814 | 0.5489353 | 0.6716322 | 0.7536077 | 600 | 2892.0000000 | 111.8771739 | -0.0004692 | -0.0006880 | 0.0016380 | 0.0035734 | 0.0015836 |
| ABLATION_NO_GOAL_MARGIN | 4884 | 0.5729322 | 0.9662491 | 0.5485258 | 0.6687604 | 0.7523040 | 600 | 2892.0000000 | 109.0500188 | -0.0007665 | -0.0011203 | 0.0012285 | 0.0007016 | 0.0002799 |
| ABLATION_NO_XG | 4884 | 0.5729839 | 0.9663293 | 0.5489353 | 0.6716423 | 0.7533348 | 0 | 2892.0000000 | 95.3172344 | -0.0007148 | -0.0010402 | 0.0016380 | 0.0035835 | 0.0013107 |
| ABLATION_NO_PROGRESSION | 4884 | 0.5727501 | 0.9659874 | 0.5501638 | 0.6720493 | 0.7535030 | 600 | 0.0000000 | 110.9260891 | -0.0009486 | -0.0013820 | 0.0028665 | 0.0039905 | 0.0014789 |

Referans tarihsel bir production sürümü değildir; güncel modelin aktif ek katmanları kapatılmış kontrollü ablation çekirdeğidir. Production contract daha yeni otoritedir; final-candidate ile hesap parametreleri aynı olduğundan dürüst performans karşılaştırması bu kontrollü koldur.

## Fold bazlı performans

| fold | test_season | model | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy | delta_vs_reference_brier_1x2 | delta_vs_reference_log_loss_1x2 | delta_vs_reference_accuracy_1x2 | delta_vs_reference_same_season_spearman | delta_vs_reference_same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2020/21 | CURRENT_PRODUCTION | 540 | 0.5348640 | 0.9132200 | 0.5944444 | 0.5830464 | 0.7153059 | -0.0002175 | -0.0002075 | 0.0000000 | 0.0025874 | 0.0009031 |
| 1 | 2020/21 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 540 | 0.5350815 | 0.9134275 | 0.5944444 | 0.5804590 | 0.7144028 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 2 | 2021/22 | CURRENT_PRODUCTION | 816 | 0.5762665 | 0.9711967 | 0.5477941 | 0.6680210 | 0.7573257 | -0.0006611 | -0.0009612 | 0.0024510 | 0.0038178 | 0.0013908 |
| 2 | 2021/22 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 816 | 0.5769276 | 0.9721579 | 0.5453431 | 0.6642033 | 0.7559349 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 3 | 2022/23 | CURRENT_PRODUCTION | 804 | 0.5922468 | 0.9933569 | 0.5149254 | 0.6503044 | 0.7533710 | -0.0012620 | -0.0018668 | 0.0049751 | 0.0061912 | 0.0012021 |
| 3 | 2022/23 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 804 | 0.5935088 | 0.9952237 | 0.5099502 | 0.6441133 | 0.7521689 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 4 | 2023/24 | CURRENT_PRODUCTION | 806 | 0.5646922 | 0.9546210 | 0.5558313 | 0.7120615 | 0.7712042 | -0.0005878 | -0.0008583 | 0.0000000 | 0.0027013 | 0.0013386 |
| 4 | 2023/24 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 806 | 0.5652800 | 0.9554793 | 0.5558313 | 0.7093602 | 0.7698656 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 5 | 2024/25 | CURRENT_PRODUCTION | 957 | 0.5694950 | 0.9613779 | 0.5642633 | 0.7184099 | 0.7715843 | -0.0004941 | -0.0007873 | 0.0041797 | 0.0014496 | 0.0007771 |
| 5 | 2024/25 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 957 | 0.5699891 | 0.9621652 | 0.5600836 | 0.7169603 | 0.7708072 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 6 | 2025/26 | CURRENT_PRODUCTION | 961 | 0.5846915 | 0.9823598 | 0.5390219 | 0.6928186 | 0.7631921 | -0.0021465 | -0.0031056 | 0.0052029 | 0.0072511 | 0.0034531 |
| 6 | 2025/26 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 961 | 0.5868380 | 0.9854653 | 0.5338189 | 0.6855675 | 0.7597391 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |

### Bağımlılığa dayanıklı pooled güven aralıkları

| metric | mean_difference | ci_95_lower | ci_95_upper | reliable_improvement |
| --- | --- | --- | --- | --- |
| brier_1x2 | -0.0009584 | -0.0015728 | -0.0002979 | True |
| log_loss_1x2 | -0.0013978 | -0.0022746 | -0.0004498 | True |

## UCL, UEL ve UECL

| model | competition | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy | delta_vs_reference_brier_1x2 | delta_vs_reference_log_loss_1x2 | delta_vs_reference_accuracy_1x2 | delta_vs_reference_same_season_spearman | delta_vs_reference_same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CURRENT_PRODUCTION | UCL | 1384 | 0.5539399 | 0.9384161 | 0.5715318 | 0.5629461 | 0.7059133 | -0.0009529 | -0.0012895 | 0.0043353 | 0.0018258 | -0.0000012 |
| CURRENT_PRODUCTION | UECL | 2073 | 0.5820429 | 0.9794461 | 0.5311143 | 0.7268177 | 0.7730841 | -0.0004971 | -0.0007468 | 0.0004824 | 0.0047553 | 0.0018243 |
| CURRENT_PRODUCTION | UEL | 1427 | 0.5774603 | 0.9731223 | 0.5578136 | 0.6811333 | 0.7366103 | -0.0016339 | -0.0024485 | 0.0056062 | 0.0047893 | 0.0015435 |

## Sıralama ve ileri sezon kontrolü

- Aynı-sezon pooled Spearman `0.672079`, pairwise `0.753503`.
- Beş forward geçişte pooled Spearman current `0.468489`, reference `0.467063`; pairwise current `0.658162`, reference `0.657936`.
- Aynı-sezon ranking diagnostiktir; forward ranking sezon sonu ratingi yalnız takip eden sezon performansına bağlar.

## Başlangıç Elo etkisi

| scope | team_seasons | unique_clubs | changed_team_seasons | changed_share | mean_abs_initial_delta | median_abs_initial_delta | p90_abs_initial_delta | p95_abs_initial_delta | maximum_positive_initial_delta | maximum_negative_initial_delta | mean_abs_rank_change | maximum_rank_gain | maximum_rank_loss | domestic_adjustment_positive_cap_hits | domestic_adjustment_negative_cap_hits | ao_first_exact_500 | ao_first_exact_2000 | minimum_current_ao_first_elo | maximum_current_ao_first_elo | changed_mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL_TEAM_SEASONS | 1887 | 504 | 1412 | 0.748278 | 5.677451 | 2.021742 | 19.936000 | 25.088952 | 30.000000 | -28.191684 | 2.133545 | 16 | -18 | 286 | 47 | 0 | 0 | 651.494994 | 1996.449185 | 7.587358 |
| 2025/26 | 236 | 236 | 181 | 0.766949 | 5.925982 | 2.311376 | 20.100000 | 24.015000 | 30.000000 | -22.700000 | 2.355932 | 16 | -16 | 39 | 6 | 0 | 0 | 810.508716 | 1994.706233 | 7.726694 |

| exposure_band | team_seasons | changed_team_seasons | mean_effective_exposure | mean_initial_delta | mean_abs_initial_delta | median_abs_initial_delta | maximum_abs_initial_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 230 | 68 | 0.000000 | 7.392861 | 7.897468 | 0.000000 | 30.000000 |
| (0,0.25] | 187 | 120 | 0.173954 | 12.239987 | 13.924119 | 17.862170 | 28.600000 |
| (0.25,0.50] | 308 | 231 | 0.379409 | 8.512979 | 9.774551 | 9.854662 | 22.460000 |
| (0.50,0.75] | 371 | 309 | 0.630636 | 3.506620 | 5.179201 | 4.428630 | 14.880000 |
| (0.75,1.00] | 791 | 684 | 0.838158 | 0.236594 | 1.720704 | 1.263157 | 6.920000 |

### En çok yükselen takım-sezonları

| season | team_name | country_code | effective_european_exposure | raw_surprise | consistency_multiplier | domestic_prior_adjustment | initial_elo_delta | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024/25 | AC Virtus | SMA | 0.00000 | 0.40143 | 0.75199 | 30.00000 | 30.00000 | 10.00000 |
| 2018/19 | KuPS Kuopio | FIN | 0.00000 | 0.30094 | 0.81398 | 30.00000 | 30.00000 | 7.00000 |
| 2018/19 | Stade Rennais | FRA | 0.00000 | 0.23158 | 0.91846 | 30.00000 | 30.00000 | 5.00000 |
| 2018/19 | Újpest TE | HUN | 0.00000 | 0.32891 | 0.86414 | 30.00000 | 30.00000 | 6.00000 |
| 2018/19 | Hapoel Haifa | ISR | 0.00000 | 0.48385 | 0.82824 | 30.00000 | 30.00000 | 1.00000 |
| 2018/19 | Tobol Kustanai | KAZ | 0.00000 | 0.25510 | 0.93577 | 30.00000 | 30.00000 | 4.00000 |
| 2018/19 | Radnicki Nis | SRB | 0.00000 | 0.20133 | 0.87927 | 30.00000 | 30.00000 | 2.00000 |
| 2018/19 | Spartak Subotica | SRB | 0.00000 | 0.39067 | 0.83266 | 30.00000 | 30.00000 | 4.00000 |
| 2018/19 | Djurgårdens IF | SWE | 0.00000 | 0.25800 | 0.95132 | 30.00000 | 30.00000 | 1.00000 |
| 2019/20 | Lechia Gdansk | POL | 0.00000 | 0.51333 | 0.73408 | 30.00000 | 30.00000 | 5.00000 |

### En çok düşen takım-sezonları

| season | team_name | country_code | effective_european_exposure | raw_surprise | consistency_multiplier | domestic_prior_adjustment | initial_elo_delta | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018/19 | Lillestrøm SK | NOR | 0.00000 | -0.19667 | 0.82915 | -28.19168 | -28.19168 | -4.00000 |
| 2024/25 | Ilves Tampere | FIN | 0.08667 | -0.35182 | 0.86848 | -30.00000 | -27.40000 | -18.00000 |
| 2020/21 | SønderjyskE | DEN | 0.13000 | -0.25476 | 0.74029 | -30.00000 | -26.10000 | -5.00000 |
| 2025/26 | Wolfsberger AC | AUT | 0.24333 | -0.30909 | 0.78487 | -30.00000 | -22.70000 | -13.00000 |
| 2019/20 | FC Utrecht | NED | 0.27000 | -0.28294 | 0.86063 | -30.00000 | -21.90000 | -8.00000 |
| 2025/26 | KA Akureyri | ISL | 0.27000 | -0.25636 | 0.84026 | -30.00000 | -21.90000 | -16.00000 |
| 2021/22 | Sileks Kratovo | MAC | 0.29333 | -0.30394 | 0.73301 | -30.00000 | -21.20000 | -10.00000 |
| 2025/26 | FC Prishtina | KOS | 0.17667 | -0.18091 | 0.84877 | -24.74698 | -20.37502 | -11.00000 |
| 2022/23 | FC Lugano | SUI | 0.27000 | -0.16444 | 0.82968 | -25.98961 | -18.97241 | -11.00000 |
| 2023/24 | Torpedo Zhodino | BLS | 0.19800 | -0.15667 | 0.86957 | -19.33391 | -15.50579 | -4.00000 |

## Sezon sonu rating etkisi

| team_seasons | changed_team_seasons | mean_abs_end_delta | median_abs_end_delta | p90_abs_end_delta | p95_abs_end_delta | maximum_positive_end_delta | maximum_negative_end_delta | mean_abs_rank_change | maximum_rank_gain | maximum_rank_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1413 | 1405 | 10.100195 | 5.708840 | 23.477094 | 31.180086 | 114.541614 | -61.186522 | 2.070771 | 15 | -22 |

### Güncel modelde en büyük sezon içi yükselişler

| season | team_name | initial_rating | end_live_rating | season_live_change | rank_change_vs_reference |
| --- | --- | --- | --- | --- | --- |
| 2020/21 | Villarreal | 1584.0857 | 2055.4568 | 471.3710 | 3.0000 |
| 2022/23 | West Ham United | 1451.4821 | 1900.0380 | 448.5559 | 0.0000 |
| 2025/26 | Nottingham Forest | 1382.3541 | 1830.3870 | 448.0329 | 9.0000 |
| 2022/23 | Lech Poznan | 1076.0726 | 1506.0973 | 430.0247 | 4.0000 |
| 2025/26 | Newcastle United | 1356.8377 | 1782.7944 | 425.9567 | 7.0000 |
| 2021/22 | Bodø/Glimt | 1138.7035 | 1551.3953 | 412.6918 | 1.0000 |
| 2024/25 | Aston Villa | 1438.0263 | 1840.7581 | 402.7318 | 2.0000 |
| 2025/26 | Aston Villa | 1715.5687 | 2115.8135 | 400.2448 | 2.0000 |
| 2025/26 | AEK Athens | 1091.1861 | 1489.2873 | 398.1012 | 4.0000 |
| 2025/26 | Crystal Palace | 1371.0186 | 1766.9024 | 395.8838 | 14.0000 |

### Güncel modelde en büyük sezon içi düşüşler

| season | team_name | initial_rating | end_live_rating | season_live_change | rank_change_vs_reference |
| --- | --- | --- | --- | --- | --- |
| 2022/23 | Olympiakos Piraeus | 1614.3665 | 1231.5552 | -382.8113 | -2.0000 |
| 2025/26 | Glasgow Rangers | 1804.1011 | 1422.0245 | -382.0766 | -6.0000 |
| 2025/26 | Maccabi Tel-Aviv | 1547.6020 | 1174.4916 | -373.1104 | -12.0000 |
| 2020/21 | AA Gent | 1499.5497 | 1151.8996 | -347.6501 | 0.0000 |
| 2025/26 | Rapid Wien | 1557.8784 | 1211.1352 | -346.7432 | -10.0000 |
| 2024/25 | RB Leipzig | 1920.1503 | 1594.6797 | -325.4706 | 0.0000 |
| 2024/25 | Qarabag FK | 1562.7972 | 1239.6947 | -323.1025 | -2.0000 |
| 2025/26 | Eintracht Frankfurt | 1883.3537 | 1563.9653 | -319.3884 | -9.0000 |
| 2021/22 | Tottenham Hotspur | 1905.1105 | 1599.6264 | -305.4841 | -3.0000 |
| 2023/24 | Ajax | 1857.6763 | 1552.9626 | -304.7137 | -1.0000 |

## Ablation: katkı hangi katmandan geliyor?

| model | brier_cost_when_disabled | log_loss_cost_when_disabled | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- |
| ABLATION_NO_DOMESTIC_SURPRISE | 0.0004892 | 0.0007098 | 0.5489353 | 0.6716322 | 0.7536077 |
| ABLATION_NO_GOAL_MARGIN | 0.0001919 | 0.0002775 | 0.5485258 | 0.6687604 | 0.7523040 |
| ABLATION_NO_XG | 0.0002436 | 0.0003577 | 0.5489353 | 0.6716423 | 0.7533348 |
| ABLATION_NO_PROGRESSION | 0.0000098 | 0.0000158 | 0.5501638 | 0.6720493 | 0.7535030 |

Pozitif `cost_when_disabled`, katman kapatıldığında loss'un yükseldiğini ve katmanın fayda sağladığını gösterir. Negatif değer katmanın mevcut kombinasyonda loss'a zarar verdiğini gösterir; bu sonuç nedensel veya bağımsız prospective kanıt değildir.

## Güvenlik ve veri kalitesi

| check | passed | observed | requirement | severity_if_failed |
| --- | --- | --- | --- | --- |
| contract_active_parameters_equal | True | 0.0000000000 | Production and final-candidate calculation parameters must match | MEDIUM |
| production_replay_exact_match | True | 0.0000000000 | Independent audit replay must match production pipeline | HIGH |
| event_match_id_unique | True | 0.0000000000 | One event row per match_id | HIGH |
| event_season_match_unique | True | 0.0000000000 | No duplicate season-match records | MEDIUM |
| identity_team_season_unique | True | 0.0000000000 | One identity row per team-season | MEDIUM |
| adjustment_team_season_unique | True | 0.0000000000 | One Domestic Surprise adjustment per team-season | MEDIUM |
| feature_team_season_unique | True | 0.0000000000 | One surprise feature row per team-season | MEDIUM |
| prediction_match_model_unique | True | 0.0000000000 | One current-model prediction per match | MEDIUM |
| probabilities_normalized | True | 0.0000000000 | 1X2 probabilities sum to one | MEDIUM |
| power_zero_sum | True | 0.0000000001 | Power Elo is conserved within season | HIGH |
| match_zero_sum | True | 0.0000000000 | Every match Power update is zero-sum | HIGH |
| progression_cap | True | 0.0000000000 | Progression bonuses remain within competition caps | MEDIUM |
| exposure_range | True | 0.0000000000 | Effective exposure stays in [0,1] | MEDIUM |
| insufficient_history_zero_adjustment | True | 0.0000000000 | Fewer than five seasons must produce zero surprise adjustment | MEDIUM |
| surprise_sign_preserved | True | 0.0000000000 | Surprise adjustment cannot reverse raw signal sign | MEDIUM |
| surprise_cap | True | 0.0000000000 | Domestic Surprise must respect +/-30 cap | MEDIUM |
| chronology_sorted | True | 0.0000000000 | event_order must be unique and UTC-monotonic within every season | MEDIUM |
| simultaneous_team_collision | True | 0.0000000000 | A team cannot appear twice at the same kickoff UTC | MEDIUM |

Bağımsız audit replay ile production pipeline eşleşmesi: `True`. En büyük expected-score farkı `0.000e+00`.

## Güçlü ve zayıf noktalar

- Güçlü: exact-date kronoloji, açık field-score/penaltı sözleşmesi, Power Elo zero-sum, bounded xG winner guard, progression cap ve eksik xG fallback denetlenebilir durumda.
- Güçlü: Domestic Surprise etkisi exposure ile otomatik azalır; beş tam sezon yoksa düzeltme sıfırdır.
- Zayıf: aktif katmanların bir kısmı manuel product kararıyla aktive edilmiştir; 2018/19-2025/26 geliştirme penceresi bağımsız prospective holdout değildir.
- Zayıf: xG yalnız 2025/26'da geniş kapsamalıdır; önceki sezonlarda current kol çoğunlukla GD fallback kullanır.
- Zayıf: progression winner-only ve non-zero-sum olduğu için AO Live toplamını artırır; Power Elo conservation geçse de görünen Live Elo toplamı korunmaz.
- Zayıf: global H sabittir; takım bazlı saha etkisi güncel production'da yoktur.

## Production kararı ve sonraki adım

**KEEP**: güncel production contract korunmalı. Bu değerlendirme contract değiştirmez. 2026/27 lig aşaması ve sonrası kilitli pre-match ledger ile prospective sonuç geldiğinde aynı paket tekrar çalıştırılmalıdır.

## Açık sorular

- xG katmanının çok sezonlu, tek sağlayıcılı per-shot veriyle katkısı aynı yönde kalacak mı?
- Progression bonusunun küçük loss katkısı prospective sıralama katkısına dönüşecek mi?
- Hiyerarşik lig-sezon + takım HomeContext, global H'yi güvenilir biçimde iyileştirebilir mi?
