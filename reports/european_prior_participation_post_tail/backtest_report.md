# European Prior Recalibration Backtest

## Kapsam

- 2018/19-2025/26 static seed evreni: 1.887 takim-sezon
- Unseen outer fold: 6
- Unseen mac: 4,884
- Aday: 81
- Production degisti: hayir

Test edilen uc eksen: European History benchmark/prior scale, giris turnuvasi
kalite proxy'si ve effective exposure cap. Turnuva kalite alani gecmis puanlarin
tek tek hangi kupada kazanildigini degil, sezon basinda bilinen entry
competition'i kullanir; bu nedenle sonuc production kaniti degil challenger
kanitidir.

## Karar

`KEEP_RESEARCH`

- Full-history aday: `b20_s1_e0.65_q1-1-1_t1_d1_p0.2`
- Brier fold kazanimi: 0/6
- Log-loss fold kazanimi: 0/6
- Pooled Brier delta: +0.000000
- Pooled log-loss delta: +0.000000
- Seed Spearman delta: +0.000000
- Seed pairwise delta: +0.000000

## Ferencvaros Diagnostigi

- Current AO First Elo: 1612.345
- Candidate AO First Elo: 1612.345
- Candidate sira: 45
- Elo farki: -0.000

## Pooled model

| model | matches | brier_1x2 | log_loss_1x2 | seed_spearman | seed_pairwise_accuracy | delta_vs_baseline_brier_1x2 | delta_vs_baseline_log_loss_1x2 | delta_vs_baseline_seed_spearman | delta_vs_baseline_seed_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CURRENT_PRODUCTION | 4884 | 0.565303 | 0.954672 | 0.469072 | 0.662981 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| NESTED_RECALIBRATION | 4884 | 0.565303 | 0.954672 | 0.469072 | 0.662981 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Fold sonuclari

| fold | test_season | selected_candidate_key | matches | brier_1x2 | baseline_brier_1x2 | delta_brier_1x2 | log_loss_1x2 | baseline_log_loss_1x2 | delta_log_loss_1x2 | accuracy_1x2 | seed_spearman | baseline_seed_spearman | delta_seed_spearman | seed_pairwise_accuracy | baseline_seed_pairwise_accuracy | delta_seed_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2020/21 | b20_s1_e0.65_q1-1-1_t1_d1_p0.2 | 540 | 0.521269 | 0.521269 | 0.000000 | 0.889104 | 0.889104 | 0.000000 | 0.618519 | 0.466595 | 0.466595 | 0.000000 | 0.664344 | 0.664344 | 0.000000 |
| 2 | 2021/22 | b20_s1_e0.65_q1-1-1_t1_d1_p0.2 | 816 | 0.573002 | 0.573002 | 0.000000 | 0.966125 | 0.966125 | 0.000000 | 0.549020 | 0.468416 | 0.468416 | 0.000000 | 0.661262 | 0.661262 | 0.000000 |
| 3 | 2022/23 | b20_s1_e0.65_q1-1-1_t1_d1_p0.2 | 804 | 0.580908 | 0.580908 | 0.000000 | 0.977129 | 0.977129 | 0.000000 | 0.524876 | 0.466037 | 0.466037 | 0.000000 | 0.662315 | 0.662315 | 0.000000 |
| 4 | 2023/24 | b20_s1_e0.65_q1-1-1_t1_d1_p0.2 | 806 | 0.557953 | 0.557953 | 0.000000 | 0.944343 | 0.944343 | 0.000000 | 0.558313 | 0.491732 | 0.491732 | 0.000000 | 0.670001 | 0.670001 | 0.000000 |
| 5 | 2024/25 | b20_s1_e0.65_q1-1-1_t1_d1_p0.2 | 957 | 0.564750 | 0.564750 | 0.000000 | 0.954548 | 0.954548 | 0.000000 | 0.561129 | 0.500181 | 0.500181 | 0.000000 | 0.673948 | 0.673948 | 0.000000 |
| 6 | 2025/26 | b20_s1_e0.65_q1-1-1_t1_d1_p0.2 | 961 | 0.577169 | 0.577169 | 0.000000 | 0.971790 | 0.971790 | 0.000000 | 0.563996 | 0.423578 | 0.423578 | 0.000000 | 0.647421 | 0.647421 | 0.000000 |

## Turnuva segmentleri

| competition | matches | brier_1x2 | baseline_brier_1x2 | delta_brier_1x2 | log_loss_1x2 | baseline_log_loss_1x2 | delta_log_loss_1x2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UCL | 1384 | 0.546330 | 0.546330 | 0.000000 | 0.926959 | 0.926959 | 0.000000 |
| UECL | 2073 | 0.577148 | 0.577148 | 0.000000 | 0.972241 | 0.972241 | 0.000000 |
| UEL | 1427 | 0.566497 | 0.566497 | 0.000000 | 0.956028 | 0.956028 | 0.000000 |

## Eksen ablation

| ablation | candidate_key | brier_1x2 | delta_vs_baseline_brier_1x2 | log_loss_1x2 | delta_vs_baseline_log_loss_1x2 | seed_spearman | delta_vs_baseline_seed_spearman | seed_pairwise_accuracy | delta_vs_baseline_seed_pairwise_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CURRENT_BASELINE | b20_s1_e0.65_q1-1-1_t1_d1_p0.2 | 0.567396 | 0.000000 | 0.957423 | 0.000000 | 0.467269 | 0.000000 | 0.661771 | 0.000000 |
| SHRINKAGE_RAISED | b20_s1_e0.65_q1-1-1_t1_d1_p0.35 | 0.567662 | 0.000266 | 0.957800 | 0.000376 | 0.466614 | -0.000655 | 0.661360 | -0.000410 |
| SELECTED | b20_s1_e0.65_q1-1-1_t1_d1_p0.2 | 0.567396 | 0.000000 | 0.957423 | 0.000000 | 0.467269 | 0.000000 | 0.661771 | 0.000000 |

## Belirsizlik

| metric | method | matches | clusters | mean_difference | ci_95_lower | ci_95_upper | reliable_improvement | reliable_harm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brier | tie_or_match | 4884 | 1612 | 0.000000 | 0.000000 | 0.000000 | False | False |
| brier | team_season | 4884 | 1413 | 0.000000 | 0.000000 | 0.000000 | False | False |
| brier | calendar_month | 4884 | 65 | 0.000000 | 0.000000 | 0.000000 | False | False |
| brier | conservative_envelope | 4884 | <NA> | 0.000000 | 0.000000 | 0.000000 | False | False |
| log_loss | tie_or_match | 4884 | 1612 | 0.000000 | 0.000000 | 0.000000 | False | False |
| log_loss | team_season | 4884 | 1413 | 0.000000 | 0.000000 | 0.000000 | False | False |
| log_loss | calendar_month | 4884 | 65 | 0.000000 | 0.000000 | 0.000000 | False | False |
| log_loss | conservative_envelope | 4884 | <NA> | 0.000000 | 0.000000 | 0.000000 | False | False |

## Ranking belirsizligi

| metric | method | folds | mean_difference | ci_95_lower | ci_95_upper | reliable_improvement | reliable_harm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed_spearman | season_block_bootstrap | 6 | 0.000000 | 0.000000 | 0.000000 | False | False |
| seed_pairwise_accuracy | season_block_bootstrap | 6 | 0.000000 | 0.000000 | 0.000000 | False | False |
