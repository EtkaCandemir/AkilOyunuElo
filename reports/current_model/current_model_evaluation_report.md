# AO European Elo Güncel Production Model Değerlendirmesi

## Teknik özet

- Qualification uyarisi: bu legacy tarihsel evaluator qualifier stage-K/retention gecisini replay etmez; asagidaki loss/ranking metrikleri yeni `0.20/0.275/0.35/0.425` davranisinin kaniti degildir. Aktif runtime davranisi production contract, dynamic engine testleri ve 2026/27 preproduction replay ile dogrulanir.
- Production/final-candidate aktif blok eşitliği: `False`; qualifier dışı çekirdek eşitliği: `True`. Qualification production override beyanı: `True`. Açıklayıcı formül alanları sayısal karşılaştırmadan çıkarılır.
- Anlamlı karşılaştırma için aynı Scale/H/K üzerinde bütün aktif ek katmanları kapalı `REFERENCE_CORE_NO_ACTIVE_EXTRAS` kolu üretildi.
- `6` unseen/development fold sezonunda `4884` maç değerlendirildi. AO rating çekirdeğinin 1X2 çıkışı Brier `0.566413`, log-loss `0.956259`, accuracy `0.5592` üretti.
- Referansa karşı farklar: Brier `-0.001640`, log-loss `-0.002915`, accuracy `+0.0047`.
- Fold kazanımları Brier `6/6`, log-loss `6/6`, aynı-sezon Spearman `6/6`, pairwise `6/6`.
- Kullanıcıya sunulan production tahmini `%50 Current ML + %50 AO Domestic Poisson (rho=0)` log-probability ensemble'dır.
- Contract'ta donmus aktivasyon kaniti olarak kayitli tarihsel nested `ML_POISSON_ENSEMBLE` kolu Brier `0.562065`, log-loss `0.949965`, accuracy `0.5612` uretmistir. Bu degerler contract'in `prediction_layer_evidence` blogundan gelir; veri yenilemesiyle tekrar kosulan `reports/production_prediction/` metrikleri degildir. Kol her fold icin Poisson kaynagi ve agirligi secer; sabit production karisiminin birebir replay olcumu degildir.
- Kararlar: rating çekirdeği **KEEP**; prediction katmanı **PROMOTE_WITH_MONITORING**. Prediction yalnız olasılık üretir, AO Live Elo'ya geri beslenmez ve sorun halinde Current AO 1X2'ye döner.

## Güncel sözleşme ve aktif mimari

- Model: `ao-european-elo-v2.0-dev-freeze`; production revision `2026-08-28-domestic-provider-season-and-fixture-integrity-fixes`; final candidate `ao-european-elo-v2.0-final-candidate-2026-08-13`.
- Statik: country benchmark `25`, European history benchmark `20`, sezon ağırlıkları `0.07/0.13/0.20/0.27/0.33`, country/european/exposure tail beta `0/0/0`.
- Domestic Prior = 500 + lig gücü bileşeni + lig/kupa başarısının lig gücüyle ölçeklenmiş bileşeni. Şampiyonluk, kupa ve duble kuralları başlangıç ratinginde kullanılır.
- Domestic Surprise aktiftir: theta `0.4`, variance penalty `0.5`, cap `+/-30.0`, tam geçmiş `5` sezon.
- Dynamic: Scale `835.561497`, H `148.544266`, K `103.980986`, carry `0.0`.
- Qualifier: base stage onemi `0.40/0.55/0.70/0.85`, delta retention `%50`, efektif K `0.20/0.275/0.35/0.425`; MAIN gecisinde reset veya mac-disi Elo degisimi yoktur.
- 1X2: normal ve iki ayaklı maçlarda draw-at-even `0.24`, tek maçta biten eleme eşleşmelerinde `0.12`; format düzeltmesi Elo state'ini değiştirmez.
- Gol farkı: alpha `0.15`, tau `300.0`, GD cap `4`.
- xG: ratio `0.3`, scale `1.25`, analitik minimum winner gain oranı `0.7`; iki taraf xG yoksa GD-only fallback.
- Progression: UCL/UEL/UECL `12/8/4`, sezon cap `48/32/16`; yalnız R16, çeyrek final, yarı final ve final, winner-only, tek tie uygulaması, sezon resetli.
- Production prediction: Current ML ve AO Domestic Poisson `0.50/0.50` log-probability blend; Poisson `rho=0`; her tahmin audit edilir; fallback Current AO 1X2'dir.
- Achievement Reserve, Competition K, Dynamic K, season carry ve takım bazlı home context aktif değildir. Ev sahibi avantajı global H olarak uygulanır.

## Baseline ve güncel model ana metrikleri

| model | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy | xg_applied_matches | total_progression_bonus | maximum_abs_match_delta | delta_vs_reference_brier_1x2 | delta_vs_reference_log_loss_1x2 | delta_vs_reference_accuracy_1x2 | delta_vs_reference_same_season_spearman | delta_vs_reference_same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CURRENT_PRODUCTION | 4884 | 0.5664125 | 0.9562592 | 0.5591728 | 0.6832577 | 0.7594208 | 2806 | 2092.0000000 | 113.7603402 | -0.0016400 | -0.0029146 | 0.0047093 | 0.0017703 | 0.0012254 |
| REFERENCE_CORE_NO_ACTIVE_EXTRAS | 4884 | 0.5680525 | 0.9591739 | 0.5544636 | 0.6814874 | 0.7581954 | 0 | 0.0000000 | 91.7317514 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| ABLATION_NO_DOMESTIC_SURPRISE | 4884 | 0.5665444 | 0.9564450 | 0.5561016 | 0.6840277 | 0.7600380 | 2806 | 2092.0000000 | 113.5718768 | -0.0015081 | -0.0027288 | 0.0016380 | 0.0025403 | 0.0018427 |
| ABLATION_NO_GOAL_MARGIN | 4884 | 0.5664738 | 0.9563382 | 0.5581491 | 0.6812141 | 0.7584290 | 2806 | 2092.0000000 | 113.7189629 | -0.0015787 | -0.0028357 | 0.0036855 | -0.0002733 | 0.0002336 |
| ABLATION_NO_XG | 4884 | 0.5670863 | 0.9572297 | 0.5581491 | 0.6823661 | 0.7584031 | 0 | 2092.0000000 | 91.4566291 | -0.0009663 | -0.0019442 | 0.0036855 | 0.0008787 | 0.0002077 |
| ABLATION_NO_PROGRESSION | 4884 | 0.5664122 | 0.9562585 | 0.5591728 | 0.6835161 | 0.7594368 | 2806 | 0.0000000 | 113.7603402 | -0.0016403 | -0.0029154 | 0.0047093 | 0.0020288 | 0.0012415 |
| ABLATION_NO_SINGLE_MATCH_DRAW | 4884 | 0.5671268 | 0.9578607 | 0.5591728 | 0.6832577 | 0.7594208 | 2806 | 2092.0000000 | 113.7603402 | -0.0009258 | -0.0013131 | 0.0047093 | 0.0017703 | 0.0012254 |
| ABLATION_NO_CUP_DOUBLE_BONUS | 4884 | 0.5664345 | 0.9563002 | 0.5579443 | 0.6837319 | 0.7597883 | 2806 | 2092.0000000 | 113.1400736 | -0.0016180 | -0.0028737 | 0.0034808 | 0.0022445 | 0.0015929 |

Bu tablodaki CURRENT_PRODUCTION etiketi geriye uyumlu replay adıdır ve AO rating çekirdeğinin olasılığını gösterir; nihai servis edilen ML+Poisson olasılığı değildir. Referans tarihsel bir production sürümü değil, aktif rating ekleri kapalı kontrollü ablation çekirdeğidir.

## Fold bazlı performans

| fold | test_season | model | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy | delta_vs_reference_brier_1x2 | delta_vs_reference_log_loss_1x2 | delta_vs_reference_accuracy_1x2 | delta_vs_reference_same_season_spearman | delta_vs_reference_same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2020/21 | CURRENT_PRODUCTION | 540 | 0.5228184 | 0.8913565 | 0.6185185 | 0.5932605 | 0.7202513 | -0.0066104 | -0.0136422 | 0.0037037 | 0.0021111 | 0.0017291 |
| 1 | 2020/21 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 540 | 0.5294288 | 0.9049987 | 0.6148148 | 0.5911494 | 0.7185222 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 2 | 2021/22 | CURRENT_PRODUCTION | 816 | 0.5740609 | 0.9676559 | 0.5477941 | 0.6716345 | 0.7580424 | -0.0005649 | -0.0009505 | -0.0012255 | 0.0003973 | 0.0003421 |
| 2 | 2021/22 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 816 | 0.5746258 | 0.9686064 | 0.5490196 | 0.6712372 | 0.7577004 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 3 | 2022/23 | CURRENT_PRODUCTION | 804 | 0.5822453 | 0.9790730 | 0.5223881 | 0.6641645 | 0.7606379 | -0.0030588 | -0.0043035 | 0.0049751 | 0.0026551 | 0.0006502 |
| 3 | 2022/23 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 804 | 0.5853041 | 0.9833764 | 0.5174129 | 0.6615094 | 0.7599877 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 4 | 2023/24 | CURRENT_PRODUCTION | 806 | 0.5594718 | 0.9464948 | 0.5583127 | 0.7250447 | 0.7789534 | -0.0002428 | -0.0009085 | 0.0000000 | 0.0016213 | 0.0021877 |
| 4 | 2023/24 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 806 | 0.5597147 | 0.9474032 | 0.5583127 | 0.7234234 | 0.7767657 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 5 | 2024/25 | CURRENT_PRODUCTION | 957 | 0.5646590 | 0.9544787 | 0.5642633 | 0.7344593 | 0.7796049 | -0.0003434 | -0.0006873 | 0.0020899 | 0.0007331 | 0.0000595 |
| 5 | 2024/25 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 957 | 0.5650024 | 0.9551660 | 0.5621735 | 0.7337262 | 0.7795454 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 6 | 2025/26 | CURRENT_PRODUCTION | 961 | 0.5787357 | 0.9739281 | 0.5619147 | 0.7031619 | 0.7706050 | -0.0010360 | -0.0012931 | 0.0166493 | 0.0031389 | 0.0022668 |
| 6 | 2025/26 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 961 | 0.5797717 | 0.9752212 | 0.5452653 | 0.7000230 | 0.7683381 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |

### Bağımlılığa dayanıklı pooled güven aralıkları

| metric | mean_difference | ci_95_lower | ci_95_upper | reliable_improvement |
| --- | --- | --- | --- | --- |
| brier_1x2 | -0.0016400 | -0.0029053 | -0.0005381 | True |
| log_loss_1x2 | -0.0029146 | -0.0052703 | -0.0009461 | True |

## UCL, UEL ve UECL

| model | competition | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy | delta_vs_reference_brier_1x2 | delta_vs_reference_log_loss_1x2 | delta_vs_reference_accuracy_1x2 | delta_vs_reference_same_season_spearman | delta_vs_reference_same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CURRENT_PRODUCTION | UCL | 1384 | 0.5500581 | 0.9323333 | 0.5758671 | 0.5632486 | 0.7066065 | -0.0025420 | -0.0048036 | 0.0028902 | -0.0045577 | -0.0026719 |
| CURRENT_PRODUCTION | UECL | 2073 | 0.5771827 | 0.9722694 | 0.5383502 | 0.7413575 | 0.7799909 | -0.0004769 | -0.0007320 | 0.0053063 | 0.0046591 | 0.0018383 |
| CURRENT_PRODUCTION | UEL | 1427 | 0.5666284 | 0.9562063 | 0.5732306 | 0.6965777 | 0.7432735 | -0.0024549 | -0.0042533 | 0.0056062 | 0.0027534 | 0.0020607 |

## Sıralama ve ileri sezon kontrolü

- Aynı-sezon pooled Spearman `0.683258`, pairwise `0.759421`.
- Beş forward geçişte pooled Spearman current `0.475864`, reference `0.472581`; pairwise current `0.661164`, reference `0.659794`.
- Aynı-sezon ranking diagnostiktir; forward ranking sezon sonu ratingi yalnız takip eden sezon performansına bağlar.

## Başlangıç Elo etkisi

| scope | team_seasons | unique_clubs | changed_team_seasons | changed_share | mean_abs_initial_delta | median_abs_initial_delta | p90_abs_initial_delta | p95_abs_initial_delta | maximum_positive_initial_delta | maximum_negative_initial_delta | mean_abs_rank_change | maximum_rank_gain | maximum_rank_loss | domestic_adjustment_positive_cap_hits | domestic_adjustment_negative_cap_hits | ao_first_exact_500 | ao_first_exact_2000 | minimum_current_ao_first_elo | maximum_current_ao_first_elo | changed_mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL_TEAM_SEASONS | 1887 | 504 | 1412 | 0.748278 | 6.582363 | 3.678304 | 19.936000 | 25.088952 | 30.000000 | -28.191684 | 2.332803 | 16 | -18 | 286 | 47 | 0 | 0 | 674.971205 | 1912.095038 | 8.796685 |
| 2025/26 | 236 | 236 | 181 | 0.766949 | 6.827313 | 3.963690 | 20.100000 | 24.015000 | 30.000000 | -22.700000 | 2.381356 | 16 | -13 | 39 | 6 | 0 | 0 | 853.093279 | 1908.028150 | 8.901911 |

| exposure_band | team_seasons | changed_team_seasons | mean_effective_exposure | mean_initial_delta | mean_abs_initial_delta | median_abs_initial_delta | maximum_abs_initial_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 230 | 68 | 0.000000 | 7.392861 | 7.897468 | 0.000000 | 30.000000 |
| (0,0.25] | 187 | 120 | 0.173954 | 12.239987 | 13.924119 | 17.862170 | 28.600000 |
| (0.25,0.50] | 308 | 231 | 0.379409 | 8.512979 | 9.774551 | 9.854662 | 22.460000 |
| (0.50,0.75] | 1162 | 993 | 0.636777 | 1.465141 | 4.294432 | 3.337112 | 14.880000 |
| (0.75,1.00] | 0 | 0 | nan | nan | nan | nan | nan |

### En çok yükselen takım-sezonları

| season | team_name | country_code | effective_european_exposure | raw_surprise | consistency_multiplier | domestic_prior_adjustment | initial_elo_delta | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024/25 | AC Virtus | SMA | 0.00000 | 0.40143 | 0.75199 | 30.00000 | 30.00000 | 5.00000 |
| 2018/19 | KuPS Kuopio | FIN | 0.00000 | 0.30094 | 0.81398 | 30.00000 | 30.00000 | 7.00000 |
| 2018/19 | Stade Rennais | FRA | 0.00000 | 0.23158 | 0.91846 | 30.00000 | 30.00000 | 3.00000 |
| 2018/19 | Újpest TE | HUN | 0.00000 | 0.32891 | 0.86414 | 30.00000 | 30.00000 | 6.00000 |
| 2018/19 | Hapoel Haifa | ISR | 0.00000 | 0.48385 | 0.82824 | 30.00000 | 30.00000 | 3.00000 |
| 2018/19 | Tobol Kustanai | KAZ | 0.00000 | 0.25510 | 0.93577 | 30.00000 | 30.00000 | 4.00000 |
| 2018/19 | Radnicki Nis | SRB | 0.00000 | 0.20133 | 0.87927 | 30.00000 | 30.00000 | 2.00000 |
| 2018/19 | Spartak Subotica | SRB | 0.00000 | 0.39067 | 0.83266 | 30.00000 | 30.00000 | 1.00000 |
| 2018/19 | Djurgårdens IF | SWE | 0.00000 | 0.25800 | 0.95132 | 30.00000 | 30.00000 | 4.00000 |
| 2019/20 | Lechia Gdansk | POL | 0.00000 | 0.51333 | 0.73408 | 30.00000 | 30.00000 | 7.00000 |

### En çok düşen takım-sezonları

| season | team_name | country_code | effective_european_exposure | raw_surprise | consistency_multiplier | domestic_prior_adjustment | initial_elo_delta | rank_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018/19 | Lillestrøm SK | NOR | 0.00000 | -0.19667 | 0.82915 | -28.19168 | -28.19168 | -8.00000 |
| 2024/25 | Ilves Tampere | FIN | 0.08667 | -0.35182 | 0.86848 | -30.00000 | -27.40000 | -13.00000 |
| 2020/21 | SønderjyskE | DEN | 0.13000 | -0.25476 | 0.74029 | -30.00000 | -26.10000 | -7.00000 |
| 2025/26 | Wolfsberger AC | AUT | 0.24333 | -0.30909 | 0.78487 | -30.00000 | -22.70000 | -10.00000 |
| 2019/20 | FC Utrecht | NED | 0.27000 | -0.28294 | 0.86063 | -30.00000 | -21.90000 | -11.00000 |
| 2025/26 | KA Akureyri | ISL | 0.27000 | -0.25636 | 0.84026 | -30.00000 | -21.90000 | -9.00000 |
| 2021/22 | Sileks Kratovo | MAC | 0.29333 | -0.30394 | 0.73301 | -30.00000 | -21.20000 | -14.00000 |
| 2025/26 | FC Prishtina | KOS | 0.17667 | -0.18091 | 0.84877 | -24.74698 | -20.37502 | -13.00000 |
| 2022/23 | FC Lugano | SUI | 0.27000 | -0.16444 | 0.82968 | -25.98961 | -18.97241 | -9.00000 |
| 2023/24 | Torpedo Zhodino | BLS | 0.19800 | -0.15667 | 0.86957 | -19.33391 | -15.50579 | -5.00000 |

## Sezon sonu rating etkisi

| team_seasons | changed_team_seasons | mean_abs_end_delta | median_abs_end_delta | p90_abs_end_delta | p95_abs_end_delta | maximum_positive_end_delta | maximum_negative_end_delta | mean_abs_rank_change | maximum_rank_gain | maximum_rank_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1413 | 1405 | 14.346159 | 8.228643 | 34.931981 | 49.916417 | 137.339171 | -66.476155 | 2.837933 | 17 | -19 |

### Güncel modelde en büyük sezon içi yükselişler

| season | team_name | initial_rating | end_live_rating | season_live_change | rank_change_vs_reference |
| --- | --- | --- | --- | --- | --- |
| 2021/22 | Feyenoord | 1277.7295 | 1718.4240 | 440.6945 | 3.0000 |
| 2025/26 | Nottingham Forest | 1382.3541 | 1813.4925 | 431.1385 | 10.0000 |
| 2020/21 | Villarreal | 1676.7507 | 2100.8242 | 424.0735 | 0.0000 |
| 2025/26 | Arsenal | 1867.4223 | 2291.3158 | 423.8935 | 0.0000 |
| 2022/23 | West Ham United | 1577.2947 | 1999.1711 | 421.8764 | 0.0000 |
| 2020/21 | Chelsea | 1832.0873 | 2245.4007 | 413.3134 | 1.0000 |
| 2022/23 | Manchester City | 1894.7375 | 2304.3122 | 409.5747 | 0.0000 |
| 2025/26 | Paris Saint-Germain | 1886.3077 | 2282.7000 | 396.3923 | 1.0000 |
| 2025/26 | Bayern München | 1882.3956 | 2277.4222 | 395.0266 | -1.0000 |
| 2022/23 | Lech Poznan | 1160.7413 | 1553.2822 | 392.5410 | 3.0000 |

### Güncel modelde en büyük sezon içi düşüşler

| season | team_name | initial_rating | end_live_rating | season_live_change | rank_change_vs_reference |
| --- | --- | --- | --- | --- | --- |
| 2022/23 | Olympiakos Piraeus | 1546.0599 | 1175.1201 | -370.9398 | -13.0000 |
| 2025/26 | Villarreal | 1770.5716 | 1448.0779 | -322.4937 | 2.0000 |
| 2020/21 | AA Gent | 1447.3664 | 1126.4840 | -320.8824 | -2.0000 |
| 2025/26 | Glasgow Rangers | 1672.8154 | 1353.0258 | -319.7896 | -4.0000 |
| 2025/26 | Maccabi Tel-Aviv | 1496.3910 | 1185.2738 | -311.1172 | -11.0000 |
| 2024/25 | Qarabag FK | 1500.5922 | 1192.9455 | -307.6467 | -3.0000 |
| 2025/26 | Eintracht Frankfurt | 1798.6970 | 1502.4058 | -296.2912 | -8.0000 |
| 2022/23 | Malmö FF | 1371.3205 | 1084.2248 | -287.0957 | -17.0000 |
| 2023/24 | Ajax | 1734.2395 | 1456.4768 | -277.7627 | -7.0000 |
| 2025/26 | Rapid Wien | 1462.4224 | 1188.5792 | -273.8432 | -4.0000 |

## Ablation: katkı hangi katmandan geliyor?

| model | brier_cost_when_disabled | log_loss_cost_when_disabled | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- |
| ABLATION_NO_DOMESTIC_SURPRISE | 0.0001319 | 0.0001858 | 0.5561016 | 0.6840277 | 0.7600380 |
| ABLATION_NO_GOAL_MARGIN | 0.0000613 | 0.0000790 | 0.5581491 | 0.6812141 | 0.7584290 |
| ABLATION_NO_XG | 0.0006737 | 0.0009705 | 0.5581491 | 0.6823661 | 0.7584031 |
| ABLATION_NO_PROGRESSION | -0.0000003 | -0.0000008 | 0.5591728 | 0.6835161 | 0.7594368 |
| ABLATION_NO_SINGLE_MATCH_DRAW | 0.0007142 | 0.0016015 | 0.5591728 | 0.6832577 | 0.7594208 |
| ABLATION_NO_CUP_DOUBLE_BONUS | 0.0000220 | 0.0000409 | 0.5579443 | 0.6837319 | 0.7597883 |

Pozitif `cost_when_disabled`, katman kapatıldığında loss'un yükseldiğini ve katmanın fayda sağladığını gösterir. Negatif değer katmanın mevcut kombinasyonda loss'a zarar verdiğini gösterir; bu sonuç nedensel veya bağımsız prospective kanıt değildir.

## Güvenlik ve veri kalitesi

| check | passed | observed | requirement | severity_if_failed |
| --- | --- | --- | --- | --- |
| contract_core_parameters_equal | True | 0.0000000000 | Production and final-candidate core calculation parameters must match | MEDIUM |
| qualification_production_override_declared | True | 0.0000000000 | Continuous qualifier retention must be an explicit production-only override | MEDIUM |
| production_replay_exact_match | True | 0.0000000000 | Independent audit replay must match production pipeline | HIGH |
| event_match_id_unique | True | 0.0000000000 | One event row per match_id | HIGH |
| event_season_match_unique | True | 0.0000000000 | No duplicate season-match records | MEDIUM |
| identity_team_season_unique | True | 0.0000000000 | One identity row per team-season | MEDIUM |
| adjustment_team_season_unique | True | 0.0000000000 | One Domestic Surprise adjustment per team-season | MEDIUM |
| feature_team_season_unique | True | 0.0000000000 | One surprise feature row per team-season | MEDIUM |
| prediction_match_model_unique | True | 0.0000000000 | One current-model prediction per match | MEDIUM |
| probabilities_normalized | True | 0.0000000000 | 1X2 probabilities sum to one | MEDIUM |
| single_match_format_draw_contract | True | 248.0000000000 | Exactly 248 frozen single-match ties use draw_at_even=0.12; all other matches use 0.24 | MEDIUM |
| power_zero_sum | True | 0.0000000001 | Power Elo is conserved within season | HIGH |
| match_zero_sum | True | 0.0000000000 | Every match Power update is zero-sum | HIGH |
| progression_cap | True | 0.0000000000 | Progression bonuses remain within competition caps | MEDIUM |
| knockout_playoff_bonus_disabled | True | 0.0000000000 | Knockout play-off cannot generate progression bonus | MEDIUM |
| exposure_range | True | 0.0000000000 | Effective exposure stays in [0,1] | MEDIUM |
| insufficient_history_zero_adjustment | True | 0.0000000000 | Fewer than five seasons must produce zero surprise adjustment | MEDIUM |
| surprise_sign_preserved | True | 0.0000000000 | Surprise adjustment cannot reverse raw signal sign | MEDIUM |
| surprise_cap | True | 0.0000000000 | Domestic Surprise must respect +/-30 cap | MEDIUM |
| chronology_sorted | True | 0.0000000000 | event_order must be unique and UTC-monotonic within every season | MEDIUM |
| simultaneous_team_collision | True | 0.0000000000 | A team cannot appear twice at the same kickoff UTC | MEDIUM |
| production_prediction_artifacts_load | True | 311.0000000000 | Checksummed ML artifact and Domestic Poisson state load; 311 AO club mappings expected after the UCL/UEL coverage expansion | HIGH |
| production_prediction_rating_feedback_disabled | True | 0.0000000000 | Served prediction cannot feed back into AO Live Elo | HIGH |

Bağımsız audit replay ile production pipeline eşleşmesi: `True`. En büyük expected-score farkı `0.000e+00`.

## Güçlü ve zayıf noktalar

- Güçlü: exact-date kronoloji, açık field-score/penaltı sözleşmesi, Power Elo zero-sum, bounded xG winner guard, progression cap ve eksik xG fallback denetlenebilir durumda.
- Güçlü: Domestic Surprise etkisi exposure ile otomatik azalır; beş tam sezon yoksa düzeltme sıfırdır.
- Zayıf: aktif katmanların bir kısmı manuel product kararıyla aktive edilmiştir; 2018/19-2025/26 geliştirme penceresi bağımsız prospective holdout değildir.
- Zayıf: xG yalnız 2025/26'da geniş kapsamalıdır; önceki sezonlarda current kol çoğunlukla GD fallback kullanır.
- Zayıf: progression winner-only ve non-zero-sum olduğu için AO Live toplamını artırır; Power Elo conservation geçse de görünen Live Elo toplamı korunmaz.
- Zayıf: KPO rota asimetrisi kaldırıldıktan sonra progression katmanının pooled loss katkısı pratikte sıfırdır; katmanın tamamen kapatılması ayrı bir ürün kararı olarak değerlendirilmelidir.
- Zayıf: global H sabittir; takım bazlı saha etkisi güncel production'da yoktur.

## Production kararı ve sonraki adım

Rating çekirdeği **KEEP** olarak korunur. Prediction katmanı **PROMOTE_WITH_MONITORING** olarak aktiftir: `%50 Current ML + %50 AO Domestic Poisson`, `rho=0`, log-probability blend, rating feedback kapalı. 2026/27 lig aşaması ve sonrası kilitli pre-match log ile AO fallback'e karşı izlenmelidir.

## Açık sorular

- xG katmanının çok sezonlu, tek sağlayıcılı per-shot veriyle katkısı aynı yönde kalacak mı?
- Progression bonusunun küçük loss katkısı prospective sıralama katkısına dönüşecek mi?
- Hiyerarşik lig-sezon + takım HomeContext, global H'yi güvenilir biçimde iyileştirebilir mi?
