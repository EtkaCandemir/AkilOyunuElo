# AO European Elo 2026/27 Prospective Holdout Protokolü

Sözleşme kanıt revizyonu: **28 Ağustos 2026**

Production revision:
`2026-08-28-domestic-provider-season-and-fixture-integrity-fixes`

Rating freeze sürümü: `ao-european-elo-v2.0-dev-freeze`

Prediction freeze sürümü: `ao-ml-poisson-ensemble-v1-production`

## 1. Kapsam Kararı

2026/27 UEFA kulüp sezonunun tamamı untouched holdout değildir. Qualifying
maçları model ve veri geliştirme sürecine temas etmiştir. Temiz prospective
değerlendirme şu kapsamla sınırlandırılır:

- UCL, UEL ve UECL lig aşaması ve sonrası.
- En erken uygun kickoff: `2026-09-08T00:00:00Z`.
- Ön eleme maçları (`Preliminary Round`, Q1/Q2/Q3 ve qualifying play-off)
  kapsam dışıdır. Buradaki play-off, `Qualifying Play-off Round` turudur.
- Lig aşamasından sonraki `Knockout round play-offs` (`KNOCKOUT_PLAYOFF`)
  kapsam içindedir; aynı kickoff ve tahmin kilidi koşullarına tabidir.
- Yalnız `generated_at_utc <= recorded_at_utc < kickoff_utc` olan kilitli
  tahminler uygundur.
- Retrospective replay hiçbir koşulda prospective kanıta dönüştürülemez.
- 2027/28 bir sonraki tam sezon holdout adayıdır.

Bu ayrım, [production contract](../contracts/ao_european_elo_v2_production.json)
`prospective_monitoring = "2026/27 league phase and later"` kapsamını açıklar.
[normalize_stage()](../src/ao_elo/dynamic.py) qualifying play-off'u `QUALIFYING`,
lig sonrası play-off'u `KNOCKOUT_PLAYOFF` olarak sınıflandırır;
[qualification_round_key()](../src/ao_elo/qualification_stage_k.py) ise ikinci
turu `MAIN` kabul eder. Ayrım
[mevcut round-mapping testiyle](../tests/test_qualification_stage_k.py) korunur.
`KNOCKOUT_PLAYOFF` turunda progression bonusu olmaması, maçın prospective
tahmin değerlendirmesinden çıkarıldığı anlamına gelmez.

## 2. Dondurulan Production Sözleşmesi

### AO First seed

```text
effective European exposure cap = 0.65
European prior katilim normalizasyonu = aktif, k = 0.20
European prior ust kuyrugu = aktif, european_tail_beta = 1.00 (kesme yok)
Kupa katkisi = aktif, w = 0.129032
Bilinmeyen lig sirasi = 0.15
Domestic Surprise = theta 0.40 / gamma 0.50 / cap +/-30
```

### Rating çekirdeği

```text
Scale = 835.5614973262
H     = 148.5442661913
K     = 103.9809863339
power carry = 0

qualifier base importance = Q1 0.40 / Q2 0.55 / Q3 0.70 / QPO 0.85
qualifier delta retention = 0.50, her qualifier maç güncellemesine gömülü
effective qualifier K     = Q1 0.20 / Q2 0.275 / Q3 0.35 / QPO 0.425
MAIN multiplier           = 1.00
MAIN entry reset/carry    = yok; maç olmadan rating değişmez

goal alpha = 0.15
goal tau   = 300
GD cap     = 4

xG ratio = 0.30
xG scale = 1.25
minimum winner gain ratio = 0.70

progression = 12 / 8 / 4
progression caps = 48 / 32 / 16
eligible = ROUND_OF_16 / QUARTERFINAL / SEMIFINAL / FINAL
```

### Base 1X2

```text
normal/two-leg draw_at_even = 0.24
single-match draw_at_even   = 0.12
draw shape                  = 1.00
```

### Kullanıcıya sunulan 1X2

```text
Current ML = LogBlend(Current AO, Structural Logistic, 0.90)
AO Poisson = LogBlend(Current AO, Domestic Poisson rho=0, 0.50)
Served 1X2 = LogBlend(Current ML, AO Poisson, 0.50)
fallback   = CURRENT_AO_1X2
rating feedback = false
```

Parametre otoritesi `contracts/ao_european_elo_v2_production.json`, runtime
artifact otoritesi `artifacts/production_prediction/manifest.json` dosyasıdır.

## 3. Operasyon Akışı

1. Sonuç içermeyen fixture doğrulanır.
2. AO Live state üzerinden Current AO 1X2 kickoff'tan önce kilitlenir.
3. Aynı pre-match feature snapshot ile production ML + Poisson tahmini üretilir.
4. AO ve served olasılıkları aynı append-only prediction loguna yazılır.
5. Maç tamamlanınca 90/120 field score ve uygun xG ile Power Elo settle edilir.
6. Tie tamamlandıysa advanced team ve progression olayı bir kez işlenir.
7. State checkpoint ve bütün audit hash'leri yazılır.

Qualifying boyunca Power state sürekli aynı kalıcı `club_id` üzerinde taşınır.
Takım turnuva değiştirirse state sıfırlanmaz. MAIN'e giriş yalnız bir aşama
sınıflandırmasıdır; ratinge ayrı bir carry kesintisi veya reset uygulanmaz.

Aynı kickoff saatindeki bütün maçlar tek pre-match snapshot kullanır. Sonuçlar
ancak tüm tahminler kaydedildikten sonra `kickoff_utc`, ardından `match_id`
sırasıyla state'e girer.

## 4. Prediction Log Sözleşmesi

Her prospective satır en az şunları saklar:

- `match_id`, sezon, kickoff ve generation UTC,
- turnuva, tur, stage, format ve neutral bilgisi,
- iki AO kulüp kimliği,
- Current AO H/D/A,
- ham ML ve Current ML H/D/A,
- ham Poisson ve AO Poisson H/D/A,
- kullanıcıya sunulan final H/D/A,
- Domestic Poisson coverage ve fallback nedeni,
- model/config/contract/manifest/ML/state SHA-256 değerleri,
- `rating_feedback_applied=false`.

Log append-only'dir. Maç sonucu geldikten sonra eski olasılık satırı overwrite
edilmez. Hash zinciri değişikliği görünür kılar; mümkünse günlük head hash'i
bağımsız ve salt okunur kayıt sisteminde yayımlanır.

## 5. Holdout Boyunca Yasak İşlemler

- Sonuca bakarak Scale, H, K, carry veya AO First parametresi değiştirmek.
- Exposure cap'i, katılım normalizasyonu `k` değerini, kupa katkısı ağırlığını
  veya bilinmeyen lig sırası değerini yeniden seçmek.
- Goal alpha/tau/cap, xG ratio/scale veya progression değerlerini değiştirmek.
- Draw intercept/shape veya single-match tanımını değiştirmek.
- ML, Poisson ve üst blend ağırlıklarını yeniden seçmek.
- Feature schema'yı sonuçtan yararlanacak biçimde genişletmek.
- Kapalı Dynamic K, Competition K veya Achievement Reserve katmanını açmak.
- Eski prediction logunu yeni modelle yeniden üretip prospective diye sunmak.

### Guard'ın kod tarafındaki karşılığı

Yukarıdaki yasakların üçü — 2026/27'yi eğitime, seçime veya cross-fitting'e
almak — artık `src/ao_elo/holdout_window.py` tarafından uygulanır. Üç artifact
üreticisi (`ml_backtest`, `domestic_poisson_backtest`, `prediction_ensemble`)
pencereyi `validate_development_window` ile doğrular: sezon dizisi
`2018/19-2025/26` ile **birebir** eşleşmeli ve hiçbir kickoff
`2026-07-01T00:00:00Z`'den sonra olmamalıdır.

Bu, önceki durumdan farklıdır. `ml_backtest` yalnız sekiz sezon **sayıyordu**;
`2019/20-2026/27` penceresi kabul ediliyor, final model 2026/27 satırlarıyla
eğitiliyor ve metadata'ya yine `2026/27_UNTOUCHED` yazılıyordu. Diğer ikisi
`seasons[-1] != "2026/27"` kontrolüyle yalnız son etikete bakıyordu; yanlış
etiketli bir satır ikisini de geçerdi.

"Dokunulmamış holdout" iddiası artık sabit metin değildir:
`untouched_holdout_label()` iddiayı yazmadan önce doğrulamayı yeniden koşar,
böylece iddia tanımladığı koşuldan uzun yaşayamaz. Bozuk pencere filtrelenmez,
**reddedilir** — sessizce satır düşürmek yanlış etiketli girdinin fark
edilmeden farklı bir eğitim kümesi üretmesine izin verirdi.

Acil yazılım bug'ı güvenlik için düzeltilmek zorundaysa yeni production revision,
artifact fingerprint ve etkilenen maç aralığı açıkça kaydedilir. Eski ve yeni
revision sonuçları tek homojen holdout gibi birleştirilmez.

## 6. Monitoring Metrikleri

### Ana tahmin metrikleri

- Standard multiclass 1X2 Brier.
- Multiclass log-loss.
- Accuracy ikincil metrik.
- Calibration slope/intercept ve ECE.
- Served ile Current AO arasındaki paired fark.

### Segmentler

- UCL, UEL ve UECL.
- Lig aşaması ve eleme aşaması.
- Single-match, two-leg ve normal format.
- Favori, dengeli ve underdog bantları.
- Domestic Poisson coverage: BOTH / ONE / NONE.
- Fallback ve non-fallback satırları.

### Rating ve veri güvenliği

- Maç Power Delta sıfır-toplam kontrolü.
- Maksimum rating hareketi ve sezon içi volatility.
- Progression cap, reset ve tek tie uygulaması.
- xG coverage, scope rejection ve GD fallback oranı.
- Artifact/schema/state hata ve fallback oranı.
- Duplicate, chronology ve team identity audit'i.

Belirsizlik tie/match, takım-sezon ve takvim ayı cluster bootstrap görünümleriyle
ayrı ayrı raporlanır. Fold sayısı bağımsız tekrar sayısı gibi yorumlanmaz.

## 7. Değerlendirme Takvimi

- Her maç: prediction ve settlement audit'i.
- Aylık: veri kalitesi, coverage, fallback ve calibration diagnostikleri.
- Lig aşaması tamamlanınca: ilk anlamlı ara rapor; parametre değişikliği yok.
- Sezon tamamlanınca: full prospective karar raporu.

Ara raporlar hata yakalamak içindir; model seçmek için kullanılmaz.

## 8. Sezon Sonu Kararları

### KEEP

Rating invariantları geçer; served model Current AO'ya karşı güvenilir zarar
göstermez ve segment/calibration davranışı kabul edilebilirdir.

### RECALIBRATE CANDIDATE

Ranking/rating sağlıklıdır fakat olasılıklarda sistematik calibration sapması
vardır. Yalnız prediction-only yeni bir sürüm, bir sonraki dönem için ayrıca
eğitilir; mevcut holdout geriye dönük yeniden yazılmaz.

### FALLBACK / ROLLBACK

Normalize olasılık, leakage, identity, chronology veya artifact güvenliği
bozulursa served katman Current AO 1X2'ye alınır. Prediction katmanı ratingden
ayrı olduğu için AO Live state'ini geri almadan güvenli rollback mümkündür.

## 9. Kanıt Yorumu

2018/19-2025/26 walk-forward sonuçları geliştirme kanıtıdır. 2026/27 sonucu ise
yalnız kickoff öncesi kilitlenmiş kayıtlarla prospective kanıt olacaktır. İki
kanıt sınıfı raporlarda açıkça ayrı tutulur.
