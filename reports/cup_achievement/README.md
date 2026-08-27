# Genelleştirilmiş Yerel Kupa Katkısı Backtest

Karar: **`KEEP_SHADOW`** → **`ACTIVE` (2026-08-27)**.

Bu paket aktivasyonun kanıtıdır ve **aktivasyondan önce** ölçülmüştür.
Katman `w = 0.129032` ile production'a alınmıştır. Aktivasyon istatistiksel
terfi değil tasarım kararıdır: nested seçim ağırlığı `0`'a çeker, yani veri
katkı eklemeyi tercih etmez. Gerekçe kuralın yapısal olarak yanlış olmasıdır.
Aktivasyon sonrası güncel metrikler `reports/current_model/` altındadır.

Bu paket `scripts/run_cup_achievement_backtest.py` çıktısının küratörlü
kopyasıdır. Ham çıktı `output/cup_achievement_backtest_2018_2026/` altındadır.

## Soru

Aktif model lig ve kupa başarısını `max` ile birleştirir:

```text
Achievement = min(cap, max(L, C) + 0.08 * L   yalnız şampiyon VE kupa ise)
```

`max` kupayı bir **taban** yapar, katkı yapmaz. Lig sıralaması kupa tabanının
(`0.62`) üzerinde olan her kupa sahibi kupasından sıfır kredi alır — 20 takımlı
bir ligde 7. sıra ve üstü. Ayrıca duble bonusu yalnız şampiyon-ve-kupa çiftinde
çalışır; 3. olup kupa kazanan hiçbir şey almaz.

Test edilen tek parametreli genelleme:

```text
Achievement = min(cap, max(L, C) + weight * min(L, C))
```

- Kupa kazanmayanda `min(L, C) = 0` → aktif modelle **birebir aynı**.
- `weight = 0` mevcut duble bonusu tamamen kaldırır (alt çapa).
- `weight = 0.129032` mevcut şampiyon-ve-kupa bonus büyüklüğünü **korur** ve
  aynı mantığı her lig+kupa kombinasyonuna genişletir.

## Sonuç

| Kapı | Sonuç |
| --- | --- |
| Yapısal doğrulama (4 kapı) | 4/4 `PASS` |
| Brier fold kazanımı | `w=0.05` ve `w=0.08` için **6/6** |
| Log-loss fold kazanımı | `w=0.05` ve `w=0.08` için **6/6** |
| Conservative envelope güvenilir iyileşme | **`False`** (hiçbir kolda) |
| Conservative envelope güvenilir zarar | `False` (hiçbir kolda) |
| Rating ekseni güvenilir iyileşme | `False` (tüm CI'lar sıfırı kesiyor) |
| Nested walk-forward seçimi | 6 fold'un **5'i `w=0`** |

Pooled Brier deltası `w=0.08` için `-0.000118`, nested seçim için `-0.000105`.

## Yorum

Mantıksal boşluk gerçek, ama kapatmak 8 sezonda **güvenilir bir iyileşme
üretmiyor**. Üç ayrı gözlem aynı yöne işaret ediyor:

1. Etkilenen kütle küçük: kupa sahipleri takım-sezonların yalnız `%19,2`'si.
2. Conservative envelope hiçbir ağırlıkta sıfırı geçmiyor.
3. Nested seçim ağırlığı `0`'a çekiyor — yani veri, kupa katkısı **eklemeyi**
   değil mevcut duble bonusu **kaldırmayı** tercih ediyor.

Üçüncü nokta bağımsız olarak ilginçtir: aktif şampiyon-ve-kupa duble bonusunun
da ölçülebilir desteği yoktur.

## Precedent karşılaştırması

`w=0.08` kolunun kanıt profili, contract'ta hâlihazırda **aktif** olan kontrollü
gol farkı katmanınınkiyle aynı sınıftadır:

| | Gol farkı (aktif) | Kupa `w=0.08` (shadow) |
| --- | --- | --- |
| Brier fold kazanımı | 6/6 | 6/6 |
| Pooled Brier delta | `-0.000174` | `-0.000118` |
| Dependency güvenilir iyileşme | `False` | `False` |
| Karar | manuel ürün kararıyla aktif | `KEEP_SHADOW` |

Bu bir tutarlılık sorusudur, salt istatistik sorusu değil. İki katman da
otomatik kapıyı geçmiyor; biri manuel kararla açılmış durumda.

## Yeniden üretim

```bash
python3 scripts/run_cup_achievement_backtest.py
```

Ön koşul: `output/domestic_surprise_variance_backtest_2018_2026/` altındaki
dondurulmuş Domestic Surprise çıktıları. Koşu production parametresi
değiştirmez; yalnız kanıt üretir.
