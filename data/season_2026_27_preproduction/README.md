# 2026/27 AO Preproduction Input Snapshot

Bu klasor production contract'ini degistirmeyen, 2026/27 sezonuna ait tarihli
bir preproduction veri snapshot'idir.

- Katilimci: 237
- Tamamlanmis UEFA maci: 342
- Yaklasan play-off fiksturu: 86
- Bes sezon domestic history tam: 195/237
- Domestic Surprise uygulanan takim: 189
- AO First Elo araligi: 837.440 - 1995.955
- Kalite kontrolleri: 17/17 PASS

Ana inputlar:

- `teams.csv`
- `country_coefficients.csv`
- `domestic_context.csv`
- `club_european_points.csv`

Replay girdileri:

- `matches_completed.csv`
- `fixtures_upcoming.csv`

## Domestic history penceresi

Her takimin bes sezonluk Domestic Surprise penceresi, kendi guncel pozisyonunun
sezonundan bir yil once biter. Iki farkli vintage vardir:

```text
QUALIFICATION_ROUTE / QUALIFICATION_CHAMPION  -> guncel 2025/26, pencere 2021-2025
digerleri (kupa rotasi, cozulemeyen)          -> guncel 2024/25, pencere 2020-2024
```

Kupa rotasi bir lig sirasi kodlamadigi icin o takimlarda en yeni cached tablo
kullanilir; bu bir sezon eskidir. Pencere onunla birlikte kaydigi icin hicbir
sezon kendi tarihsel ortalamasinin icinde yer almaz. Aksi halde surprise, o
sezonun kendi agirligi kadar (`0.33`) mekanik olarak sonumlenirdi.

2025/26 yerel lig tablolari kaynak cache'te bulunmadigi icin bu takimlarin
pozisyonu bir sezon eski kalir; audit dosyasindaki `current_position_vintage`
kolonu bunu acikca kaydeder ve tahmin edilerek doldurulmaz.

Eksik domestic history tahmin edilmez. Bes tam sezonu olmayan takimda aktif
pipeline Domestic Surprise'i `INSUFFICIENT_HISTORY` olarak sifirlar. Iki tarafli
ve zaman kapsami dogrulanmis xG henuz bulunmadigi icin tamamlanmis 2026/27
maclari `GOAL_MARGIN_ONLY` fallback'iyle isaretlenmistir.
