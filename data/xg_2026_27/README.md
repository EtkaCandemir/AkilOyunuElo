# AO UEFA 2026/27 xG Veri Seti

Canli sezonun xG kaynagi. Kabul sozlesmesi `data/xg_2020_2026/` ile aynidir:
kimlik dogrulamasi, skorun birebir eslesmesi, kronoloji ve kapsam denetimi.
Eksik xG doldurulmaz.

## Kapsam

| Turnuva | Tur | Mac | Uygun | Oran |
|---|---|---:|---:|---:|
| UCL | Q1 | 28 | 2 | 7.1% |
| UCL | Q2 | 28 | 6 | 21.4% |
| UCL | Q3 | 20 | 8 | 40.0% |
| UEL | Q1 | 12 | 4 | 33.3% |
| UEL | Q2 | 18 | 10 | 55.6% |
| UEL | Q3 | 26 | 12 | 46.2% |
| UECL | Q1 | 52 | 0 | 0.0% |
| UECL | Q2 | 98 | 22 | 22.4% |
| UECL | Q3 | 60 | 24 | 40.0% |
| **Toplam** | | **342** | **88** | **25.7%** |

Kapsam tur ilerledikce artar; buyuk kulupler geç turlarda devreye girer. Bu,
tarihsel on eleme ortalamasi olan `%11`'in ustundedir.

## Baglanti

Bu veri `data/season_2026_27_preproduction/matches_completed.csv` icindeki
`xg_home`, `xg_away`, `xg_analysis_eligible` ve `xg_fallback` kolonlarina
kopyalanir; replay xG'yi oradan okur. `tests/test_xg_source_wiring.py`
kopyalanan degerlerin kaynakla birebir ayni oldugunu pinler.

## Yeniden uretim

Bu sezon `exact_date_events.csv` icinde bulunmadigi icin events dosyasi
preproduction paketinden turetilir:

```bash
python3 scripts/build_2025_26_xg_dataset.py \
  --season 2026/27 --events <turetilmis_events.csv> \
  --output-root data/xg_2026_27
```

## Lisans

Kamuya acik erisim yeniden dagitim lisansi vermez.
