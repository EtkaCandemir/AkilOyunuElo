# 2026/27 Holdout Ledger Hardening

**Tarih:** 2026-08-30  
**Kapsam:** prediction ledger guard'ları, 396 prediction'ın doğru zamanla
revision 1 olarak kaydı, ledger provenance/anchor, model explainer coverage ve
contract/config türetimi  
**Model davranışı:** değişmedi  
**Production state SHA-256:**
`62ffabe7bd78ed59844aa6e09c93e235c960de5d41453c4aaa51aa199c05ee8d`
(başlangıç ve bitiş aynı)

## Sonuç

Ledger'ın doğrulanmış F1/F2/F3/F6 kusurları kapatıldı. Mevcut 396 revision 0
satırı silinmedi veya değiştirilmedi; doğru zaman semantiği taşıyan 396 revision
1 satırı zincirin üstüne eklendi. Güncel zincir 792 prediction, 0 settlement ve
`valid=true` durumunda. Anchor güncel head, dosya SHA ve sayaçlarla eşleşiyor.

Model parametresi, production contract veya prediction artifact'ı değiştirilmedi.

## Test kapıları

| Kapı | Sonuç |
| --- | --- |
| Başlangıç tam paket | `1240 passed`, 1 warning, 277.87 s |
| Düzeltme öncesi ledger/PDF karşı örnekleri | `22 failed, 53 passed` |
| Builder gerçek-saat testi, düzeltme öncesi | `1 failed` (`resolve_generated_at` yoktu) |
| Ledger guard paketi | `75 passed` |
| Ledger + PDF + preproduction hedefli paket | `90 passed` |
| Bitiş tam paket | `1297 passed`, 1 warning, 275.07 s |
| Dar ruff kontrolü | `All checks passed` |
| `git diff --check` | temiz |

Tek pytest uyarısı joblib'in fiziksel çekirdek sayısını okuyamamasıydı; aynı
uyarı başlangıç paketinde de vardı. `build_2026_27_prediction_features.py`
dosyasının tam ruff taramasında değişiklik öncesinden kalan dört B905 ve bir B007
bulgusu görünür; proje kararı gereği B905 toplu düzeltilmedi. Değişen ledger,
PDF üretici ve test yüzeyindeki dar ruff kontrolü temizdir.

## 1. Prediction zamanı

**Kod:** `src/ao_elo/prediction_ledger.py:300`,
`scripts/build_2026_27_prediction_features.py:85`

**Önce:** Ledger `generated_at < kickoff` ve `recorded_at < kickoff` kurallarını
ayrı ayrı uyguluyor, fakat `generated_at > recorded_at` satırını kabul ediyordu.
Builder CLI zamanı verilmezse en erken kickoff eksi altı saat şeklinde sentetik
bir gelecek zamanı yazıyordu.

**Sonra:** Ledger şu tek nedensellik kuralını uygular:

```text
generated_at_utc <= recorded_at_utc < kickoff_utc
```

Builder'ın varsayılanı prediction çağrısından hemen önce okunan gerçek UTC
saatidir. CLI override korunmuştur. Gerçek saati seçme gerekçesi, mevcut
operasyon komutunun parametresiz ve otomasyonda kullanılabilmesi; sentetik zaman
üretmemesidir. Açık CLI değeri verilirse servis UTC/timezone guard'ları yine
çalışır.

**Kanıtlayan testler:**

- `test_a_row_generated_after_it_was_recorded_is_refused`
- `test_a_row_generated_exactly_when_recorded_is_accepted`
- `test_prediction_builder_default_generated_at_is_the_real_utc_clock`

İlk test düzeltme öncesinde satırı kabul ettiği için kırıldı. Builder testi de
gerçek-saat resolver'ı eklenmeden önce kırıldı.

## 2. Atomik append, ön-doğrulama ve process kilidi

**Kod:** `src/ao_elo/prediction_ledger.py:483-564`

**Önce:** read/revision seçimi lock dışında yapılıyor, canlı dosya doğrudan
append modunda yazılıyor ve mevcut zincir ancak yazımdan sonra doğrulanıyordu.
İki process aynı sequence/head'i seçebiliyor; yarım yazım ve önceden bozuk zincir
canlı dosyada ilave byte bırakıyordu.

**Sonra:** Şu işlemlerin tamamı kalıcı sibling lock inode'u üzerinde POSIX
exclusive `flock` altında gerçekleşir:

```text
read -> mevcut zinciri doğrula -> revision seç -> batch'i doğrula
     -> aynı dizinde temp dosyaya yaz -> flush + fsync
     -> temp zinciri tam doğrula -> atomic os.replace
```

Publish öncesi hata temp dosyayı siler ve canlı ledger'ı byte-identical bırakır.
Lock dosyaları `.gitignore` kapsamındadır. Reader'lar atomik replace nedeniyle ya
eski ya yeni tam dosyayı görür.

**Kanıtlayan testler:**

- `test_publish_failure_leaves_the_live_ledger_byte_identical`
- `test_partial_temp_write_failure_leaves_the_live_ledger_byte_identical`
- `test_an_invalid_existing_chain_is_rejected_before_any_write`
- `test_two_processes_cannot_select_the_same_sequence_and_head`

İlk, üçüncü ve dördüncü karşı örnekler eski kodda kırıldı. ENOSPC testi temp
yazımının yarısında kontrollü `OSError` üretir; yeni kodda canlı byte'lar ve
geçerli head değişmez.

## 3. Settlement doğrulaması ve prediction bağı

**Kod:** `src/ao_elo/prediction_ledger.py:387-481`

**Önce:** Kickoff'tan önce settlement, negatif/fractional/non-finite skor ve
skorla çelişen outcome kabul ediliyordu. Eşdeğer UTC offset'i taşıyan kickoff
string karşılaştırması yüzünden reddediliyordu. Settlement hangi prediction
revision'ına bağlandığını taşımıyordu.

**Sonra:** Bütün batch yazımdan önce şu guard'lardan geçer:

- `recorded_at_utc > kickoff_utc`,
- iki gol finite, negatif olmayan exact integer,
- `outcome` field score'dan türetilen `HOME/DRAW/AWAY` ile birebir aynı,
- kickoff UTC normalize edilerek; iki kulüp kimliği birebir karşılaştırılarak
  fixture identity doğrulanır,
- payload, kilitli tahminin `prediction_entry_hash` ve `ledger_revision`
  değerini taşır.

**Kanıtlayan testler:**

- `test_settlement_recorded_at_or_before_kickoff_is_refused`
- `test_settlement_goals_must_be_finite_nonnegative_integers`
- `test_settlement_outcome_must_match_the_field_score`
- `test_settlement_kickoff_identity_is_compared_as_normalized_utc`
- `test_settlement_explicitly_binds_the_locked_prediction`

Bu karşı örneklerin tamamı eski kodda beklenen noktalarda kırıldı.

## 4. Tam 40 kolonluk prediction şeması

**Kod:** `src/ao_elo/prediction_ledger.py:51-96`  
**Test:** `tests/test_prediction_ledger.py`

`REQUIRED_PREDICTION_FIELDS` içine şu sekiz servis alanı eklendi:

```text
ml_home_probability
ml_draw_probability
ml_away_probability
poisson_home_probability
poisson_draw_probability
poisson_away_probability
poisson_component_fallback
fallback_reason
```

Test fixture'ı artık doğrudan
`PRODUCTION_PREDICTION_LOG_COLUMNS` sabitinin 40 kolonundan türetilir. Parametrik
test her kolonu tek tek siler. Eski kod yalnız yukarıdaki sekiz eksiltmeyi kabul
etti; düzeltmeden sonra 40/40 eksiltme reddedilir.

## 5. Revision 1 ve provenance

Builder çıktısı append öncesi ölçüldü:

| Ölçüm | Sonuç |
| --- | --- |
| Satır / kolon | 396 / 40, source schema birebir |
| Kupa dağılımı | UCL 144 / UEL 144 / UECL 108 |
| Earliest / latest kickoff | 2026-09-08 16:45Z / 2027-01-28 20:00Z |
| Prediction status | 396 `ACTIVE_ENSEMBLE` |
| Fallback | 0 |
| Imputed model input | 31 |
| Coverage | BOTH 338 / ONE 54 / NONE 4 |

Revision 1 zamanları:

```text
generated_at_utc = 2026-08-30T13:01:03.262137+00:00
recorded_at_utc  = 2026-08-30T13:01:39.138475+00:00
earliest kickoff = 2026-09-08T16:45:00+00:00
```

396/396 satır `generated <= recorded < kickoff` kuralını sağladı. Revision 1
payload'ları ile served CSV arasında 40 kolon boyunca hücre farkı `0` bulundu.

Append-only koruma ölçümleri:

- revision 0: 396 satır,
- revision 1: 396 satır,
- ilk 396 ham satırın SHA-256 değeri:
  `12e333ababf6c837a9f0cb72561244257c12236780c5c7295a77944742c921fb`
  (append öncesi dosya SHA ile aynı),
- eski head:
  `bc09379e45df2a8ff2462bf1dbe7358ef47e38f7df6b856274c5ea6d145f7985`,
- yeni head:
  `a0debe686ac202ca6103044b5f6c95a629fd54d9960751fb69022357fbcc3dbc`,
- güncel ledger dosya SHA-256:
  `164052913feeb01f71da3469b537cb098f91f30d1bb486a52fdf0cae0a25c785`.

`data/prediction_ledger/PROVENANCE.md`, revision 0'ın neden değerlendirme dışı
olduğunu, neden silinmediğini ve revision 1'in kilitli holdout tahmini olduğunu
kaydeder. `ledger_anchor_2026_27.json` güncel head, file SHA ve 792/792/0
sayaçlarıyla birebir eşleşir.

## 6. PDF coverage ve türetilen hücreler

**Kod:** `scripts/build_model_explainer_pdf.py:75`, `:646`, `:719`  
**Belge:** `docs/AO_EUROPEAN_ELO_V2_MODEL_EXPLAINER.md`

Coverage metni gerçek runtime davranışıyla eşleştirildi:

- `ONE`: mevcut tarafın profili kullanılır, eksik taraf nötrdür;
- `NONE`: Poisson bileşeni AO tabanına düşer.

Gerçek 396 satırda ölçülen maksimum `|raw Poisson - AO|`: `ONE=0.163576`,
`NONE=0.000000` (`BOTH=0.189896`).

Katman tablosundaki Domestic Surprise cap, Structural ML efektif ağırlığı ve
Domestic Poisson efektif ağırlığı artık sırasıyla aktif config ve contract'tan
hesaplanır. Bellek içi test config/contract değerlerini değiştirerek hücrelerin
`±17`, `0,32` ve `0,18` değerlerine hareket ettiğini doğrular; production
contract ile üretilen PDF `±30`, `0,45`, `0,25` gösterir.

PDF yeniden üretildi:
`docs/pdf/AkilOyunu_Elo_v2_Ayrintili_Model_Anlatimi.pdf`. On iki sayfanın kontakt
sayfası, ayrıca değişen 8. ve 9. sayfalar tam çözünürlükte görsel olarak kontrol
edildi; taşma, kesilme veya okunamaz tablo bulunmadı.

## Yapılmayanlar ve ölçüm sınırları

- Model parametresi, `contracts/` veya `artifacts/` altında dosya değiştirilmedi.
  Domestic production state SHA başlangıç ve bitişte aynıdır.
- Git commit veya push yapılmadı. Bu nedenle güncel anchor henüz bağımsız bir
  dış zaman tanığına bağlanmış değildir.
- Gerçek ledger'da settlement yoktur. Settlement guard'ları sentetik fixture ve
  kontrollü karşı örneklerle ölçüldü; provider status/extra-time/shootout sonucu
  entegrasyonu ölçülmedi.
- Disk gerçekten doldurulmadı; ENOSPC Python write sınırında fault injection ile
  üretildi. Process kill'in tam `os.replace` anı, kernel/power-loss dayanıklılığı,
  NFS/network filesystem ve çok-host lock davranışı ölçülmedi. POSIX `flock`
  aynı yerel filesystem üzerindeki iki fork process ile ölçüldü.
- Hash zincirinin tek başına zaman damgasını kanıtlamadığı bilinen sınır
  değişmedi.
- Çalışma sırasında iki ilgisiz untracked dosya göründü:
  `scripts/build_ao_first_elo_pdf.py` ve
  `docs/pdf/AO_2026_27_AO_First_Elo_Tam_Dokuman.pdf`. Bu çalışma onları üretmedi,
  okumada/duzenlemede kullanmadı ve değiştirmedi.
