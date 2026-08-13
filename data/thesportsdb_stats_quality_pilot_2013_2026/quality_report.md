# TheSportsDB Tarihsel İstatistik Kalite Pilotu

## Karar

**STOP_RICH_STATS_COLLECTION**

- Örnek maç: 600
- Kimlik doğruluğu: 100.0%
- Core şut kapsamı: 26.7%
- xG kapsamı: 18.3%
- Mevcut xG içindeki şüpheli placeholder: 30.0%

Bu pilot yalnız çok sezonlu zengin feature toplamaya başlanıp başlanmayacağını
belirler. Sonuçlar AO Elo veya ML production katmanını otomatik değiştirmez.

## Dönem ve ülke kapsamı

`coverage_by_country_era.csv` her ülke ve dönem için kimlik, şut ve xG kapsamını
ayrı gösterir. Kapsam eşiğini geçmeyen strata sonraki toplamada dışarıda bırakılır.
