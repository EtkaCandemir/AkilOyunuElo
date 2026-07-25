# AO European Elo 2026/27 Prospective Holdout Protokolü

Sözleşme revizyonu: 23 Temmuz 2026

Model freeze tarihi: 20 Temmuz 2026

## Kapsam Kararı

2026/27 UEFA kulüp sezonunun tamamı untouched holdout değildir. UCL 7 Temmuz,
UEL 9 Temmuz ve UECL 7 Temmuz 2026'da başlamıştır; bunların bir bölümü model
freeze tarihinden önce oynanmıştır. Sonuçları bilinen maçlar daha sonra replay
edilerek prospective kanıta dönüştürülemez.

Temiz 2026/27 holdout kapsamı:

- UCL, UEL ve UECL lig aşaması ve sonraki maçlar.
- En erken başlangıç: `2026-09-08T00:00:00Z`.
- Qualifying ve play-off maçları kapsam dışıdır.
- Yalnızca `generated_at_utc < kickoff_utc` olan kilitler uygundur.
- `run_dynamic_elo.py` replay çıktıları uygun değildir.
- 2027/28, bir sonraki tam sezon prospective holdout adayıdır.

UEFA'nın yayımladığı 2026 takviminde UCL ilk lig aşaması maçları 8-10 Eylül,
UEL ilk maçları 16-17 Eylül ve UECL ilk maçı 15 Ekim 2026 olarak yer alır:

<https://www.uefa.com/news-media/news/02a0-1f71bdf70a9a-b6067bd647f2-1000--2026-european-football-calendar-match-and-draw-dates-for-a/>

## Operasyon Akışı

1. Sonuç içermeyen tek satırlık `fixtures.csv` hazırlanır.
2. `run_dynamic_live.py lock` maç başlamadan çalıştırılır.
3. Tahmin, üretim zamanı, kullanılan state ve config kimliğiyle append-only
   `pre_match_log.csv` ledger'ına yazılır.
4. Sonuç kesinleşince ayrı `matches.csv` satırı hazırlanır.
5. `run_dynamic_live.py settle` aynı kilidi doğrular, rating'i günceller ve tam
   state checkpoint'i kaydeder.

Gol farkı karşılaştırma kolları production state'inden tamamen ayrı çalıştırılır:

1. Lig aşaması katılımcılarının tam AO First Elo dosyası kesinleşince
   `run_goal_difference_shadow.py initialize` ile bağımsız shadow state kurulur.
2. Her production `lock` işleminden sonra aynı fixture için shadow `lock`
   çalıştırılır.
3. Sonuç geldiğinde production `settle` sonrasında shadow `settle` çalıştırılır.
4. Shadow ledger ve update dosyaları production rating'ini veya checkpoint'ini
   hiçbir koşulda değiştiremez.

`PRE_SPECIFIED` production'daki aktif `0.10/300` koludur. Diğer kollar ve
gol farksız `BASE` yalnız karşılaştırma amaçlıdır:

```text
BASE             alpha=0.000, tau=300
PRE_SPECIFIED    alpha=0.100, tau=300
PRIOR_GRID_BEST  alpha=0.200, tau=400
EXTENDED_BEST    alpha=0.125, tau=800
```

Bu kollar holdout başladıktan sonra değiştirilemez. Yeni bir aday ancak ayrı bir
araştırma kohortu olarak eklenebilir; mevcut holdout ile birleştirilemez.

Eşzamanlı maçlar kickoff öncesinde aynı state snapshot'ından ayrı ayrı
kilitlenebilir. Sonuçlar `kickoff_utc`, sonra `match_id` artan sırasıyla settle
edilir. Kilitten sonra ilgili takım rating'i değişmişse kayıt geçersiz sayılır.

Ledger SHA-256 hash zinciri eski satır değişikliğini görünür kılar. Hash zinciri
harici güvenilir zaman damgası değildir. Daha güçlü denetim için gün sonu ledger
head hash'i salt okunur bir release, commit veya bağımsız kayıt sisteminde
yayımlanmalıdır.

## Değerlendirme Kilidi

Holdout boyunca sonuçlara bakılarak Scale, H, K, carry, tail beta, goal margin,
reserve veya turnuva çarpanı değiştirilemez. UCL/UEL/UECL ve zero-exposure
segmentleri raporlanabilir, fakat ara sonuçlar parametre seçimi için kullanılamaz.

Kontrollü `0.10/300` gol farkı katmanı manuel kararla production'da aktiftir.
En az `300` uygun prospective maç ve bunların en az `75` UCL maçı tamamlanınca
zorunlu izleme incelemesi yapılır. Katmanın korunması için:

- Brier ve log-loss nokta farkları iyileşme yönünde olmalıdır.
- Dependency zarfı güvenilir zarar göstermemelidir.
- Pooled forward Spearman ve pairwise farkı negatif olmamalıdır.
- Tek bir fold'da Spearman gerilemesi `0.005`, pairwise gerilemesi `0.0025`
  değerini aşmamalıdır.
- Gözlenen maksimum gol çarpanı `1.30` güvenlik sınırını aşmamalıdır.
- Katmanı kapatma veya parametre değiştirme otomatik değil, sürüm değişikliği
  ve yeni config fingerprint ile manuel yapılmalıdır.

Bu eşikler 2026/27 başlamadan önce kaydedilmiş prospective izleme kapılarıdır.
Ara sonuçlara bakılarak `alpha`, `tau` veya cap değiştirilemez. Güvenilir zarar
görülürse production geri alma kararı ayrıca kaydedilir.

Aktif olasılık katmanı standart H/D/A çıktısı üretir. Holdout raporunda:

- Ana tahmin metrikleri standart üç sınıflı 1X2 Brier ve log-loss'tur.
- Legacy expected-score MSE yalnız tarihsel karşılaştırma diagnostikleri için
  ayrı etiketle korunur.
- Spearman ve pairwise sonuçları cross-fitted rakip/saha düzeltilmiş sıralama
  hedefiyle raporlanır.
- Güven aralıkları tie/match, takım-sezon ve takvim ayı bootstrap görünümleriyle
  ayrı ayrı ve en geniş zarfla duyarlılık analizi olarak raporlanır.
