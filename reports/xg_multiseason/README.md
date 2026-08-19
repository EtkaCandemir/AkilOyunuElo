# Bounded xG Katmani: Alti Sezonluk Yeniden Dogrulama

Karar: **kanit guclendi, katman degismedi.** Acik urun onayiyla contract.in
`xg_performance_evidence` alani guncellenmistir; katman parametreleri aynidir.

Ham cikti `output/xg_multiseason_backtest_2020_2026/` altindadir.

## Soru

Aktif contract `xg_performance` katmanini tek sezonluk kanitla acti:

```text
full_season_matches      961
xg_eligible_matches      606
manual_product_decision  true
```

O karar verildiginde yalniz 2025/26 xG'si vardi; onceki yedi sezon
`GOAL_MARGIN_ONLY` fallback'iyle oynandi ve katman hakkinda hicbir sey
soyleyemedi. FotMob xG 2020/21'e kadar uzandigi icin uygun orneklem
`606` -> `2827` seviyesine cikti.

## Sonuc

| Kol | Brier | Log-loss |
|---|---:|---:|
| `XG_SIX_SEASON` | **0.573732** | **0.966444** |
| `XG_2025_26_ONLY` | 0.574160 | 0.967043 |
| `NO_XG` | 0.574346 | 0.967315 |

Conservative envelope, alti sezonluk kol icin **her segmentte** guvenilir
iyilesme veriyor:

| Segment | Brier farki | %95 CI | Guvenilir |
|---|---:|---:|:--:|
| ALL | -0.000614 | [-0.001149, -0.000108] | evet |
| PHASE:MAIN | -0.001169 | [-0.002103, -0.000210] | evet |
| XG_PRESENT | -0.001385 | [-0.002465, -0.000278] | evet |

Log-loss'ta da ucu birden gecer.

## Kazanc nerede olusuyor

| Segment | Mac | Brier farki |
|---|---:|---:|
| Ana asama | 3257 | -0.001169 |
| On eleme | 3083 | -0.000028 |
| xG var | 2827 | **-0.001385** |
| xG yok | 3513 | +0.000006 |

Kazanc tam olarak xG'nin bulundugu yerde. Bulunmadigi yerde etki sifira yakin,
yani katman bilgi yokken gurultu eklemiyor. Tasarimin amaclanan davranisi budur.

## Sinirlar

- Sezon dagilimi duzensiz: alti xG'li sezonun dordunde kazaniyor, `2023/24` ve
  `2024/25`'te hafif kaybediyor. Havuzlanmis kazancin buyuk kismi `2022/23`'ten
  gelir (`-0.002362`).
- Bu yeni bir katman degil. `xg_performance` zaten aktiftir; kosu yalnizca
  contract'in tek sezonluk kanitini genisletir.
- Asil fark modelden degil **veriden** gelir: sevk edilen kapsamla `-0.000186`,
  alti sezonluk kapsamla `-0.000614`.
- `2026/27`'nin bugune kadar oynanmis her maci on elemedir ve orada kapsam
  `%11`'dir; kazanc lig asamasi baslayinca devreye girer.

## Uygulanan contract guncellemesi

Acik urun onayiyla yapildi. Katman ayni kaldi; yalnizca kanit kaydi degisti:

```text
onceki : 961 mac / 606 uygun, tek sezon, manual_product_decision
simdiki: 4884 mac / 2827 uygun, alti sezon, conservative envelope reliable
```

`production_revision: 2026-08-19-xg-evidence-revalidated-six-seasons`

Degisen ust duzey alanlar yalniz `decisions`, `production_revision` ve
`xg_performance_evidence`'dir. `xg_performance`, `dynamic_core` ve
`prediction_layer` bloklari birebir aynidir. Ilk aktivasyon kaydi
`superseded_evidence` altinda provenance icin korunur.

## Yeniden uretim

```bash
python3 scripts/run_xg_multiseason_backtest.py
```

Onkosul: `data/xg_2020_2026/` veri seti ve dondurulmus Domestic Surprise
ciktilari. Kosu production parametresi degistirmez.
