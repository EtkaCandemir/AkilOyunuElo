# xG Bilgili Gol Beklentisi

Karar: **`KEEP_SHADOW`.** Production contract'a dokunulmadi.

## Soru

Skor katmani `Diagnostic` statusundedir cunku climatology'yi zar zor gecer:
ust 2.5'te `0.0018`, KG var'da `0.0003` Brier. Ayrica her iki orani yalniz Elo
farkindan turetir, takim formu hic yoktur.

Bu kosu her iki tarafa birer form terimi ekler ve ayni yapiyi iki kaynakla
besler. `GOALS` kolu kontroldur: onsuz `XG` kolundaki kazanc, xG'nin mi yoksa
sadece form terimi eklemenin mi katkisi oldugu ayirt edilemezdi.

## Sonuc

| Kol | exact_score_nll | Elo'ya fark |
|---|---:|---:|
| `XG` | **3.027008** | **-0.002834** |
| `GOALS` | 3.029642 | -0.000201 |
| `ELO_ONLY` | 3.029843 | 0 |

xG, gol formunu **4 fold'un 4'unde** de gecer.

### Conservative envelope

| Kol | Segment | Fark | %95 CI | Guvenilir |
|---|---|---:|---:|:--:|
| `XG` | `XG_PRESENT` | -0.006970 | [-0.011187, -0.002668] | **evet** |
| `XG` | `ALL` | -0.002834 | [-0.007159, +0.001017] | hayir |
| `GOALS` | `XG_PRESENT` | -0.001142 | [-0.005266, +0.002893] | hayir |
| `GOALS` | `ALL` | -0.000201 | [-0.004052, +0.003589] | hayir |

Gol kontrolunde hicbir segment guvenilir degildir; kazanc xG'ye ozgudur.

### Segment yapisi

| Segment | XG farki |
|---|---:|
| Ana asama | -0.008186 |
| xG var | -0.006970 |
| On eleme | +0.003243 |
| xG yok | +0.003065 |

xG'nin olmadigi yerde kol hafifce zarar verir: model bir katsayi tasir ama
girdi sifir gelir. Katman bu yuzden **kosullu** olmalidir.

## Iki onemli sinir

**1. Reponun mevcut en iyi kolunu gecmez.**

| | exact_score_nll | orneklem |
|---|---:|---|
| Bu kosunun `XG` kolu | 3.027008 | 3528 |
| Repo `DOMESTIC_ATTACK_DEFENCE_POISSON` | **3.016594** | 4884 |

Orneklemler ayni degildir, dogrudan kiyaslanamaz. Ancak buradaki taban
(`ELO_ONLY`) reponun kolundan daha basittir. Gosterilen sey "xG form terimi
basit bir tabana eklendiginde yardim eder"dir; "xG mevcut en iyi modele
eklendiginde yardim eder" **gosterilmemistir**.

**2. Turev pazarlar gecmez.** Ust 2.5 Brier `-0.001123`, CI `[-0.002555,
+0.000275]` sifiri keser. Skor dagilimi duzelir, ust/alt ve KG var climatology
sinirinda kalir.

## Siradaki adim

xG form terimini `DOMESTIC_ATTACK_DEFENCE_POISSON` uzerine koymak, onun yerine
degil. Bu, skor katmanini `Diagnostic`'ten cikarmak icin gereken tek kritik
testtir.

## Yeniden uretim

```bash
python3 scripts/run_xg_goal_expectation_backtest.py
```

Onkosul: `data/xg_2020_2026/` ve `run_current_model_evaluation.py` ciktilari.
