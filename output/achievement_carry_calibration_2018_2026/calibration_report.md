# AO Achievement Reserve and Season Carry Calibration

Decision: **PROVISIONAL_ACCEPT_POWER_CARRY_TROPHY_REJECTED**

## Scope

Seasons: 2018/19 through 2025/26; outer folds: 6.
Global identity audit: 1887 team-season rows and 
506 persistent clubs.
Match results update only zero-sum Power Elo. Final winners receive a separate
non-zero-sum Achievement Reserve. Power and reserve can influence the next
season only through independently controlled carry and decay parameters.

```text
Power_start = (1-carry) * AO_First_current + carry * Power_end_previous
Reserve_start = decay * Reserve_end_previous
AO_live = Power + Reserve
Trophy_bonus = UCL_base * competition_reference  # capped reserve
```

Competition references are fixed at UCL=1.00, UEL=0.65 and UECL=0.45 in this
run. They are not calibrated unless the reserve layer first proves useful.

## Fold Selections

| Fold | Unseen | Selected carry | Selected decay | Selected trophy | Achievement carry | Achievement decay | Achievement trophy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2020/21 | 0.85 | 0 | 0 | 0.85 | 0.25 | 20 |
| 2 | 2021/22 | 1 | 0 | 0 | 1 | 0.25 | 20 |
| 3 | 2022/23 | 0.75 | 1 | 20 | 0.75 | 1 | 20 |
| 4 | 2023/24 | 0.9 | 0 | 0 | 0.9 | 0.25 | 20 |
| 5 | 2024/25 | 0.9 | 0 | 0 | 0.9 | 0.25 | 20 |
| 6 | 2025/26 | 0.85 | 0 | 0 | 0.85 | 0.25 | 20 |

## Unseen Comparison

| Model | Fold wins | Mean Brier difference | CI lower | CI upper |
| --- | ---: | ---: | ---: | ---: |
| Nested selected | 6/6 | -0.002543 | -0.003927 | -0.001235 |
| Forced achievement | 6/6 | -0.002542 | -0.003928 | -0.001237 |
| Carry only | 6/6 | -0.002612 | -0.004017 | -0.001286 |
| Reference .5/.5/40 | 6/6 | -0.002063 | -0.002748 | -0.001411 |

## Isolated Trophy Contribution

The achievement challenger is compared directly with the fold-selected
carry-only challenger so carry improvement cannot be attributed to trophies.

Trophy beat matched carry-only in **1/6** folds.
Incremental Brier difference: 0.000070 
(95% CI -0.000015 to 0.000155).

| Competition | Matches | Trophy vs carry Brier difference | Log-loss difference |
| --- | ---: | ---: | ---: |
| UCL | 1384 | 0.000080 | 0.000173 |
| UECL | 2073 | 0.000036 | 0.000067 |
| UEL | 1427 | 0.000109 | 0.000253 |

## Competition Segments

| Model | Competition | Matches | Brier difference | Log-loss difference |
| --- | --- | ---: | ---: | ---: |
| Forced achievement | UCL | 1384 | -0.002862 | -0.007317 |
| Forced achievement | UECL | 2073 | -0.001155 | -0.002830 |
| Forced achievement | UEL | 1427 | -0.004246 | -0.010126 |
| Reference | UCL | 1384 | -0.002941 | -0.007072 |
| Reference | UECL | 2073 | -0.001071 | -0.002501 |
| Reference | UEL | 1427 | -0.002654 | -0.006279 |

## Parameter Stability

| Parameter | Mode | Fold frequency | Range |
| --- | ---: | ---: | ---: |
| selected_power_carry | 0.85 | 2/6 | 0.75-1 |
| selected_reserve_decay | 0 | 5/6 | 0-1 |
| selected_ucl_trophy_base | 0 | 5/6 | 0-20 |
| carry_only_power_carry | 0.85 | 3/6 | 0.85-1 |
| achievement_power_carry | 0.85 | 2/6 | 0.75-1 |
| achievement_reserve_decay | 0.25 | 5/6 | 0.25-1 |
| achievement_ucl_trophy_base | 20 | 6/6 | 20-20 |

## Full-Data Results

All-candidate selection: `carry=0.85`, 
`decay=0`, 
`UCL trophy=0`; 
Brier=0.161285.
Best forced achievement: `carry=0.85`, 
`decay=0.25`, 
`UCL trophy=20`; 
Brier=0.161293.

| Competition | Reference | Trophy bonus | Decay | Reserve cap |
| --- | ---: | ---: | ---: | ---: |
| UCL | 1 | 20 | 0.25 | 80 |
| UEL | 0.65 | 13 | 0.25 | 80 |
| UECL | 0.45 | 9 | 0.25 | 80 |

## Observed Competition Strength

This table describes the pre-match AO Live Rating distribution of actual
participants. It measures the strength of the field, not a prestige bonus.

| Phase | Competition | Team-match observations | Mean | Median | IQR |
| --- | --- | ---: | ---: | ---: | --- |
| KNOCKOUT | UCL | 516 | 894.1 | 896.1 | 854.9-932.5 |
| KNOCKOUT | UEL | 796 | 805.4 | 802.2 | 772.3-840.8 |
| KNOCKOUT | UECL | 450 | 748.2 | 750.2 | 713.7-777.7 |
| MAIN_STAGE | UCL | 1728 | 829.6 | 828.3 | 775.3-882.0 |
| MAIN_STAGE | UEL | 2016 | 747.3 | 742.9 | 705.6-787.3 |
| MAIN_STAGE | UECL | 1008 | 701.0 | 695.9 | 656.8-743.8 |
| QUALIFYING | UCL | 1376 | 691.1 | 692.0 | 630.4-746.3 |
| QUALIFYING | UEL | 2102 | 655.5 | 648.9 | 606.0-698.0 |
| QUALIFYING | UECL | 2688 | 648.6 | 642.2 | 610.0-678.8 |

## External Elo Benchmark

The paired AO ablation remains valid without an external Elo because every
candidate uses the same matches and initial ratings. An external historical Elo
would still improve absolute validation and expose initial-rating bias. It cannot
be joined safely until exact match dates and pre-match historical snapshots are
available; using current or end-of-season Elo would introduce look-ahead leakage.
