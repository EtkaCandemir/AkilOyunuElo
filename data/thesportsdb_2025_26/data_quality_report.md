# TheSportsDB 2025/26 Veri Kalitesi Raporu

## Veri ve Grain

- `events.csv`: 961 maç, bir satır bir AO maçı.
- `event_stats_long.csv`: 11664 istatistik kaydı.
- `event_timeline.csv`: 14151 olay kaydı.
- `event_lineups.csv`: 17510 kadro kaydı.

## Kimlik ve Tutarlılık

- Eşleşen maç: 961/961.
- İkinci aşama ad/tarih fallback: 6.
- AO ile TheSportsDB saha skoru uyuşmazlığı: 3.
- Provider event ID ve AO `match_id` ilişkisi bire birdir.

## Kapsam

- Stats: 648/961 (67.4%).
- Ham xG alani: 648/961 (67.4%).
- Supheli xG yer tutucu: 211.
- Analize uygun TheSportsDB xG: 437/961 (45.5%).
- FotMob ile ortak analize uygun mac: 437.
- Ortak orneklem toplam xG MAE: 0.3037; toplam xG bias (TheSportsDB - FotMob): -0.0506.
- Timeline: 938/961 (97.6%).
- Lineup: 845/961 (87.9%).

## Risk ve Kullanım Kararı

- **High:** Skor uyuşmazlıkları analiz öncesi maç bazında incelenmelidir; provider skoru AO sonucunun üzerine yazılmamıştır.
- **High:** xG süre ve penaltı kapsamı belgelenmemiştir; TheSportsDB xG production girdisi olarak otomatik kabul edilmemelidir.
- **High:** Sifir xG ile pozitif sut sayisi veya iki tarafli sifir xG bulunan provider satirlari ham olarak korunmus, analizden dislanmistir.
- **Medium:** FotMob karsilastirmasi kaynaklar arasi sistematik farki olcer; iki kaynaktan hangisinin gercek saha kalitesine daha yakin oldugunu tek basina kanitlamaz.
- **Medium:** Stats/timeline/lineup eksikliği ağırlıklı olarak ön eleme segmentlerinde beklenebilir; `coverage_summary.csv` aşama bazında kanıttır.
- **Low:** Event detail alanlarındaki medya, açıklama ve hava durumu değerleri seyrek olabilir; ham biçimde korunmuştur.
