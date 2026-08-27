# 2026/27 AO Preproduction Input Snapshot

Bu klasor production contract'ini degistirmeyen, 2026/27 sezonuna ait tarihli
bir preproduction veri snapshot'idir.

- Katilimci: 237
- Tamamlanmis UEFA maci: 342
- Yaklasan play-off fiksturu: 86
- Bes sezon domestic history tam: 193/237
- Domestic Surprise uygulanan takim: 191
- CW rotasi current-table eslesmesi: 33/36
- AO First Elo araligi: 837.440 - 1995.955
- Kalite kontrolleri: 22/22 PASS

Ana inputlar:

- `teams.csv`
- `country_coefficients.csv`
- `domestic_context.csv`
- `club_european_points.csv`

Replay girdileri:

- `matches_completed.csv`
- `fixtures_upcoming.csv`

Denetim girdileri:

- `domestic_history_audit.csv`
- `cw_domestic_evidence_audit.csv`
- `data_quality_audit.csv`
- `source_manifest.json`

## Domestic standings sozlesmesi

Tum takimlar icin 2026/27 Avrupa katilimini belirleyen son tamamlanmis yerel
sezon kullanilir. Sonbahar-ilkbahar liglerinde bu `2025/26`, takvim yili
liglerinde tamamlanmis `2025` sezonudur. Kassiesa qualification snapshot'inin
isaret ettigi lig sayfasi checksum'li cache'e alinir; `CW` rotasi artik eski
`2024/25` tablosuna dusmez.

Guncel ust lig tablosunda bulunmayan kupa sampiyonlarina eski veya tahmini bir
pozisyon verilmez. Bu takimlar audit dosyasinda acik `N/A` olarak kalir. Yerel
ligi olmayan Liechtenstein takimlari da ayni sekilde `NO_DOMESTIC_LEAGUE`
olarak kaydedilir.

Eksik domestic history tahmin edilmez. Bes tam sezonu olmayan takimda aktif
pipeline Domestic Surprise'i `INSUFFICIENT_HISTORY` olarak sifirlar. Iki tarafli
ve zaman kapsami dogrulanmis xG bulunan 88 mac korunur; diger
tamamlanmis maclar `GOAL_MARGIN_ONLY` fallback'iyle isaretlenir. Statik sezon
verisini yeniden uretmek daha once dogrulanmis xG'yi silmez; match identity ve
saha skoru degismisse build acik hata verir.
