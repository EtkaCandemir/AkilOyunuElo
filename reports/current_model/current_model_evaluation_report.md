# AO European Elo Güncel Production Model Değerlendirmesi

## Teknik özet

- Qualification uyarisi: bu legacy tarihsel evaluator qualifier stage-K/retention gecisini replay etmez; asagidaki loss/ranking metrikleri yeni `0.20/0.275/0.35/0.425` davranisinin kaniti degildir. Aktif runtime davranisi production contract, dynamic engine testleri ve 2026/27 preproduction replay ile dogrulanir.
- Production contract hesap parametreleri final-candidate ile eşittir: `False`. Birebir JSON eşitliği `False`; fark production'a sonradan eklenen açıklayıcı formül alanlarıdır.
- Anlamlı karşılaştırma için aynı Scale/H/K üzerinde bütün aktif ek katmanları kapalı `REFERENCE_CORE_NO_ACTIVE_EXTRAS` kolu üretildi.
- `6` unseen/development fold sezonunda `4884` maç değerlendirildi. AO rating çekirdeğinin 1X2 çıkışı Brier `0.571537`, log-loss `0.963594`, accuracy `0.5500` üretti.
- Referansa karşı farklar: Brier `-0.002162`, log-loss `-0.003775`, accuracy `+0.0027`.
- Fold kazanımları Brier `6/6`, log-loss `6/6`, aynı-sezon Spearman `5/6`, pairwise `6/6`.
- Kullanıcıya sunulan production tahmini `%50 Current ML + %50 AO Domestic Poisson (rho=0)` log-probability ensemble'dır: Brier `0.568093`, log-loss `0.959242`, accuracy `0.5538`.
- Kararlar: rating çekirdeği **KEEP**; prediction katmanı **PROMOTE_WITH_MONITORING**. Prediction yalnız olasılık üretir, AO Live Elo'ya geri beslenmez ve sorun halinde Current AO 1X2'ye döner.

## Güncel sözleşme ve aktif mimari

- Model: `ao-european-elo-v2.0-dev-freeze`; production revision `2026-08-19-xg-evidence-revalidated-six-seasons`; final candidate `ao-european-elo-v2.0-final-candidate-2026-08-13`.
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
| CURRENT_PRODUCTION | 4884 | 0.5715371 | 0.9635941 | 0.5499590 | 0.6731068 | 0.7540802 | 2806 | 2092.0000000 | 116.6536914 | -0.0021616 | -0.0037753 | 0.0026618 | 0.0050480 | 0.0020561 |
| REFERENCE_CORE_NO_ACTIVE_EXTRAS | 4884 | 0.5736987 | 0.9673694 | 0.5472973 | 0.6680588 | 0.7520241 | 0 | 0.0000000 | 94.8104820 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| ABLATION_NO_DOMESTIC_SURPRISE | 4884 | 0.5720130 | 0.9642947 | 0.5493448 | 0.6735018 | 0.7545130 | 2806 | 2092.0000000 | 117.0426313 | -0.0016857 | -0.0030747 | 0.0020475 | 0.0054430 | 0.0024890 |
| ABLATION_NO_GOAL_MARGIN | 4884 | 0.5716881 | 0.9638030 | 0.5493448 | 0.6705579 | 0.7530089 | 2806 | 2092.0000000 | 114.6594310 | -0.0020106 | -0.0035664 | 0.0020475 | 0.0024991 | 0.0009848 |
| ABLATION_NO_XG | 4884 | 0.5723343 | 0.9647240 | 0.5485258 | 0.6715864 | 0.7533108 | 0 | 2092.0000000 | 95.3172344 | -0.0013644 | -0.0026454 | 0.0012285 | 0.0035276 | 0.0012867 |
| ABLATION_NO_PROGRESSION | 4884 | 0.5715371 | 0.9635937 | 0.5499590 | 0.6732795 | 0.7540402 | 2806 | 0.0000000 | 116.6536914 | -0.0021617 | -0.0037757 | 0.0026618 | 0.0052207 | 0.0020162 |
| ABLATION_NO_SINGLE_MATCH_DRAW | 4884 | 0.5721937 | 0.9652135 | 0.5499590 | 0.6731068 | 0.7540802 | 2806 | 2092.0000000 | 116.6536914 | -0.0015051 | -0.0021559 | 0.0026618 | 0.0050480 | 0.0020561 |
| ABLATION_NO_CUP_DOUBLE_BONUS | 4884 | 0.5714817 | 0.9635257 | 0.5501638 | 0.6733553 | 0.7541518 | 2806 | 2092.0000000 | 116.6536914 | -0.0022170 | -0.0038437 | 0.0028665 | 0.0052966 | 0.0021278 |

Bu tablodaki CURRENT_PRODUCTION etiketi geriye uyumlu replay adıdır ve AO rating çekirdeğinin olasılığını gösterir; nihai servis edilen ML+Poisson olasılığı değildir. Referans tarihsel bir production sürümü değil, aktif rating ekleri kapalı kontrollü ablation çekirdeğidir.

## Fold bazlı performans

| fold | test_season | model | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy | delta_vs_reference_brier_1x2 | delta_vs_reference_log_loss_1x2 | delta_vs_reference_accuracy_1x2 | delta_vs_reference_same_season_spearman | delta_vs_reference_same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2020/21 | CURRENT_PRODUCTION | 540 | 0.5281868 | 0.8987230 | 0.5944444 | 0.5860811 | 0.7165619 | -0.0068947 | -0.0147045 | 0.0000000 | 0.0056221 | 0.0021591 |
| 1 | 2020/21 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 540 | 0.5350815 | 0.9134275 | 0.5944444 | 0.5804590 | 0.7144028 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 2 | 2021/22 | CURRENT_PRODUCTION | 816 | 0.5758061 | 0.9703008 | 0.5465686 | 0.6688348 | 0.7574245 | -0.0011215 | -0.0018571 | 0.0012255 | 0.0046315 | 0.0014896 |
| 2 | 2021/22 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 816 | 0.5769276 | 0.9721579 | 0.5453431 | 0.6642033 | 0.7559349 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 3 | 2022/23 | CURRENT_PRODUCTION | 804 | 0.5898924 | 0.9899682 | 0.5223881 | 0.6517275 | 0.7540741 | -0.0036164 | -0.0052554 | 0.0124378 | 0.0076142 | 0.0019052 |
| 3 | 2022/23 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 804 | 0.5935088 | 0.9952237 | 0.5099502 | 0.6441133 | 0.7521689 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 4 | 2023/24 | CURRENT_PRODUCTION | 806 | 0.5644940 | 0.9537829 | 0.5483871 | 0.7144983 | 0.7724999 | -0.0007859 | -0.0016964 | -0.0074442 | 0.0051381 | 0.0026342 |
| 4 | 2023/24 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 806 | 0.5652800 | 0.9554793 | 0.5558313 | 0.7093602 | 0.7698656 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 5 | 2024/25 | CURRENT_PRODUCTION | 957 | 0.5693697 | 0.9611045 | 0.5621735 | 0.7166840 | 0.7712263 | -0.0006194 | -0.0010608 | 0.0020899 | -0.0002764 | 0.0004191 |
| 5 | 2024/25 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 957 | 0.5699891 | 0.9621652 | 0.5600836 | 0.7169603 | 0.7708072 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |
| 6 | 2025/26 | CURRENT_PRODUCTION | 961 | 0.5849805 | 0.9829940 | 0.5400624 | 0.6932334 | 0.7634490 | -0.0018575 | -0.0024713 | 0.0062435 | 0.0076659 | 0.0037099 |
| 6 | 2025/26 | REFERENCE_CORE_NO_ACTIVE_EXTRAS | 961 | 0.5868380 | 0.9854653 | 0.5338189 | 0.6855675 | 0.7597391 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 | 0.0000000 |

### Bağımlılığa dayanıklı pooled güven aralıkları

| metric | mean_difference | ci_95_lower | ci_95_upper | reliable_improvement |
| --- | --- | --- | --- | --- |
| brier_1x2 | -0.0021616 | -0.0034426 | -0.0010507 | True |
| log_loss_1x2 | -0.0037753 | -0.0062098 | -0.0017100 | True |

## UCL, UEL ve UECL

| model | competition | matches | brier_1x2 | log_loss_1x2 | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy | delta_vs_reference_brier_1x2 | delta_vs_reference_log_loss_1x2 | delta_vs_reference_accuracy_1x2 | delta_vs_reference_same_season_spearman | delta_vs_reference_same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CURRENT_PRODUCTION | UCL | 1384 | 0.5521906 | 0.9345519 | 0.5708092 | 0.5608314 | 0.7050608 | -0.0027022 | -0.0051538 | 0.0036127 | -0.0002890 | -0.0008537 |
| CURRENT_PRODUCTION | UECL | 2073 | 0.5815684 | 0.9787530 | 0.5306319 | 0.7290698 | 0.7738741 | -0.0009716 | -0.0014399 | 0.0000000 | 0.0070074 | 0.0026144 |
| CURRENT_PRODUCTION | UEL | 1427 | 0.5757283 | 0.9697399 | 0.5578136 | 0.6829873 | 0.7375210 | -0.0033660 | -0.0058310 | 0.0056062 | 0.0066433 | 0.0024542 |

## Sıralama ve ileri sezon kontrolü

- Aynı-sezon pooled Spearman `0.673107`, pairwise `0.754080`.
- Beş forward geçişte pooled Spearman current `0.470420`, reference `0.467063`; pairwise current `0.658813`, reference `0.657936`.
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
| 1413 | 1405 | 14.203887 | 7.828325 | 35.072705 | 49.134795 | 131.420651 | -65.692198 | 2.738854 | 15 | -22 |

### Güncel modelde en büyük sezon içi yükselişler

| season | team_name | initial_rating | end_live_rating | season_live_change | rank_change_vs_reference |
| --- | --- | --- | --- | --- | --- |
| 2020/21 | Villarreal | 1584.0857 | 2096.8250 | 512.7393 | 4.0000 |
| 2022/23 | West Ham United | 1451.4821 | 1946.0325 | 494.5504 | 3.0000 |
| 2025/26 | Nottingham Forest | 1382.3541 | 1824.7564 | 442.4023 | 9.0000 |
| 2022/23 | Lech Poznan | 1076.0726 | 1514.5682 | 438.4956 | 5.0000 |
| 2021/22 | Bodø/Glimt | 1138.7035 | 1564.4100 | 425.7065 | 4.0000 |
| 2022/23 | Fenerbahçe | 1204.8857 | 1626.9920 | 422.1063 | 6.0000 |
| 2024/25 | Aston Villa | 1438.0263 | 1860.0322 | 422.0059 | 3.0000 |
| 2021/22 | Feyenoord | 1300.3385 | 1722.0062 | 421.6677 | 5.0000 |
| 2024/25 | Tottenham Hotspur | 1589.9783 | 2006.7140 | 416.7357 | 1.0000 |
| 2025/26 | Newcastle United | 1356.8377 | 1772.5041 | 415.6663 | 6.0000 |

### Güncel modelde en büyük sezon içi düşüşler

| season | team_name | initial_rating | end_live_rating | season_live_change | rank_change_vs_reference |
| --- | --- | --- | --- | --- | --- |
| 2022/23 | Olympiakos Piraeus | 1614.3665 | 1197.7687 | -416.5979 | -4.0000 |
| 2025/26 | Glasgow Rangers | 1804.1011 | 1422.0245 | -382.0766 | -5.0000 |
| 2025/26 | Maccabi Tel-Aviv | 1547.6020 | 1174.4916 | -373.1104 | -12.0000 |
| 2020/21 | AA Gent | 1499.5497 | 1143.7704 | -355.7793 | -1.0000 |
| 2023/24 | Ajax | 1857.6763 | 1506.8955 | -350.7808 | -7.0000 |
| 2025/26 | Rapid Wien | 1557.8784 | 1211.1352 | -346.7432 | -10.0000 |
| 2024/25 | Qarabag FK | 1562.7972 | 1224.5992 | -338.1980 | -5.0000 |
| 2025/26 | Eintracht Frankfurt | 1883.3537 | 1563.9653 | -319.3884 | -9.0000 |
| 2022/23 | Malmö FF | 1393.2777 | 1076.7551 | -316.5226 | -11.0000 |
| 2024/25 | RB Leipzig | 1920.1503 | 1604.0365 | -316.1138 | 0.0000 |

## Ablation: katkı hangi katmandan geliyor?

| model | brier_cost_when_disabled | log_loss_cost_when_disabled | accuracy_1x2 | same_season_spearman | same_season_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- |
| ABLATION_NO_DOMESTIC_SURPRISE | 0.0004758 | 0.0007006 | 0.5493448 | 0.6735018 | 0.7545130 |
| ABLATION_NO_GOAL_MARGIN | 0.0001510 | 0.0002089 | 0.5493448 | 0.6705579 | 0.7530089 |
| ABLATION_NO_XG | 0.0007972 | 0.0011299 | 0.5485258 | 0.6715864 | 0.7533108 |
| ABLATION_NO_PROGRESSION | -0.0000001 | -0.0000004 | 0.5499590 | 0.6732795 | 0.7540402 |
| ABLATION_NO_SINGLE_MATCH_DRAW | 0.0006565 | 0.0016194 | 0.5499590 | 0.6731068 | 0.7540802 |
| ABLATION_NO_CUP_DOUBLE_BONUS | -0.0000554 | -0.0000684 | 0.5501638 | 0.6733553 | 0.7541518 |

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
| production_prediction_artifacts_load | True | 171.0000000000 | Checksummed ML artifact and Domestic Poisson state load; 171 AO club mappings expected | HIGH |
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
