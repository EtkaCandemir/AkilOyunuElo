# Domestic provider sezonu ve fikstür bütünlüğü onarımı

Tarih: **2026-08-28**. Production revision:
`2026-08-28-domestic-provider-season-and-fixture-integrity-fixes`.

## Sonuç ve eski raporun düzeltmesi

Eski checkpoint canlı başlangıç state'i olarak kullanılamazdı: **LIT'in 2026
provider sezonundan 61 fikstür history ve live girdilerinde iki kez işleniyordu.**
Yalnız eski tarihli 20 satırı öne çıkarmak asıl sorunu eksik anlatıyordu.
Birleşik girdi **71 çift fikstür / 142 gözlem** içeriyordu; LIT bunların
**61 çift / 122 gözlem** kısmıdır.

Yalnız tekrarları ve üç doğrulanmış skor çelişkisini gideren **tanısal** replay,
311 eligible kulübün **71'inin** girdisini değiştirdi. Bu tanısal dedup üretim
çözümü olarak kullanılmadı. Provider sezonu ve coverage seçimi de düzeltilerek
kaynaklardan yapılan tam onarımda **81 kulübün** girdisi değişti
(`max_abs_input_delta > 1e-12`). Bunlar fixture sayısı değil kulüp sayısıdır.

Yeni production checkpoint: **75.226 history + 1.101 live = 76.327 maç,
34 lig, 0 kanonik fikstür tekrarı**. 312 kimlikten 311'i eligible; hedef
UCL/UEL evreni 79/80 kaldı. Torreense'nin kapsam kapısı değiştirilmedi.
Son işlenmiş kickoff **2026-08-20 19:00 UTC**. Bu onarım mevcut cache'i kullandı;
28 Ağustos'a kadar yeni canlı sonuçlar çekildiği anlamına gelmez.

Bağımsız taze replay, yayımlanan checkpoint'in `engine_state` alanıyla **tam
eşit** çıktı. Eski birleşik girdi artık hem merge hem replay girişinde reddediliyor.

Kanıtlar:

- [Sayım ve replay doğrulaması](domestic_integrity_fix_2026_08_28/state_change_summary.json)
- [311 kulübün önce/sonra girdileri](domestic_integrity_fix_2026_08_28/eligible_club_state_changes.csv)
- [Eski 142 gözlem](domestic_integrity_fix_2026_08_28/before_duplicate_observations.csv)
- [Ülke bazında tarihsel satır değişimi](domestic_integrity_fix_2026_08_28/historical_counts_by_country.csv)

## Uygulanan sıra

1. **#4 — kaynak sezonu:** Secondary `provider_season`/`season` varsa kaynak
   alanı kullanılır. Takvim liglerinde yoksa UTC takvim yılı kullanılır;
   AO'nun Temmuz sınırından provider yılı türetilmez. Tarih filtresi de provider
   sezonuna uygulanır. LIT 2026 satırları artık 2013–2025 tarihsel kabulüne giremez.
   Kaynak sezonu bulunmayan kış liglerinde Temmuz sınırı tahmini ayrıca belgelenmiştir.
2. **Fikstür guard'ı:** Alias çözümü merge'den önce yapılır. Kanonik lig,
   ev/deplasman takım anahtarları ve normalize UTC aynıysa yeni event ID ikinci
   maç yaratmaz. Merge, replay ve engine batch preflight bunu reddeder;
   başarısız batch state'i değiştirmez. Runtime'da otomatik dedup yoktur.
3. **Kaynak uzlaştırması:** Üç çelişkili skor için aşağıdaki resmî kararlar
   versiyonlandı. Aynı fikstür/aynı skor gözlemlerinin çıkarılması ayrıca
   KEEP/REMOVE_OBSERVATION audit'i üretir. Bilinmeyen skor çelişkisi build'i
   durdurur; `drop_duplicates` ile skor seçilmez, ham cache değiştirilmez.
4. **#5 — coverage:** Mod yalnız gerçekten tekrar ediyorsa kullanılır; bütün
   sayımlar tekilse `ceil(median(counts))` kullanılır. `%95` eşiği değişmez.
   `format_expectation_method` audit edilir; bu resmi format doğrulaması değildir.
5. **Yeniden üretim:** Veri, historical/live checkpoint, ML feature store ve
   değerlendirmeleri, 54 adaylı Poisson yüzeyi, nested ensemble, genişletilmiş
   coverage kanıtı, external benchmark, current evaluation/snapshot, 2026/27
   replay ve tahmin feature çıktıları, ilgili docs/ai ve dört PDF yenilendi.

### Skor kaynakları

| Fikstür | Doğrulanmış skor | Tutulan / çıkarılan TSD event ID | Birincil kanıt |
| --- | --- | --- | --- |
| AE Larissa–AEK, 2021-02-15 | **2–4** | 1068057 / 1091502 | [AEK resmî maç haberi](https://www.aekfc.gr/newsdetails/aaa-217-126092.htm?lang=el&path=-2068606472) |
| Taraz–Kaisar, 2020-10-17 | **2–2** | 1039304 / 1039365 | [QFL resmî 13. hafta protokolü](https://qfl.kz/news/protokolnye-dannye-13-go-tura-olimpbet-chempionata-kazakhstana-5806/) |
| Alanyaspor–Sivasspor, 2021-01-31 | **3–1** | 1049070 / 1084686 | [TFF maç protokolü](https://www.tff.org/Default.aspx?macId=219183&pageID=29) |

Kararlar [domestic_fixture_reconciliations.json](../data/domestic_fixture_reconciliations.json)
içindedir. Yeni/değişmiş gözlem mevcut karar kümesine uymuyorsa otomatik
genişletilmez. Kaynak kararı skoru doğrular; her provider kickoff'unun bağımsız
doğrulandığı iddia edilmez. 10 tarihsel fazla gözlem için [base audit](../data/domestic_league_matches_2013_2026/fixture_reconciliation_audit.csv)
ve [expansion audit](../data/domestic_league_expansion_ucl_uel/fixture_reconciliation_audit.csv) saklanır.

## Kapsam etkisi: kazanılan ve kaybedilen sezonlar

Kaynak sezonu düzeltmesi kapsamı net olarak artırdı, ama iki lig-sezonunu da
kapsam kapısının dışına çıkardı. Önceki revision ile satır bazında karşılaştırma:

| Değişim | Lig-sezon | Önce | Sonra |
| --- | --- | ---: | ---: |
| Kazanç | KAZ 2015 / 2016 / 2017 | 0 | 192 / 192 / 198 |
| Kazanç | GEO 2019 | 100 | 180 |
| Kazanç | LIT 2013 | 64 | 144 |
| Kayıp | GEO 2014 | 120 | **0** |
| Kayıp | LIT 2020 | 60 | **0** |
| Taşıma | LIT 2026 | 97 | 0 (live dosyasına geçti) |

Net etki `+956` KAZ, `+80` GEO, `+80` LIT kazanç; `-120` ve `-60` kayıp.

İki kayıp da coverage kapısındandır: GEO 2014 `120/184 = 0.652`,
LIT 2020 `60/127 = 0.472`. İkisi de secondary arşivde tam olarak vardır ve
oynanmıştır. Bu sezonlar önceki revision'da **yanlışlıkla** dahildi; bozuk
provider-sezon türetmesi onları geçerli bir sezon kovasına karıştırıyordu.
Etiket düzeltilince kapı tasarlandığı gibi çalıştı.

Yayımlanan `league_season_quality.csv` bu iki satırı
`REJECTED / UNAVAILABLE / FETCH_OR_PARSE_ERROR:ValueError` olarak gösteriyor.
Etiket yanıltıcı: primary kaynağın hatasını taşıyor, secondary'nin coverage
reddini değil. Kaynak seçimi `(country, provider_season)` başına tek satır
yazdığı için secondary'nin gerçek gerekçesi audit'te korunmuyor. Bu ayrı bir
audit-izi kusurudur ve bu turda kapatılmadı.

Mevcut kanıtla "gerçekten kısa sezon" ile "eksik arşiv" ayırt edilemez. Beklenen
sayı elle pinlenmedi; boşluk `docs/ai/DATA_CONTRACTS.md` §11.2 altında **bilinen
ve kabul edilmiş kapsam boşluğu** olarak kayda geçirildi. Bu iki sezon
2026/27'ye 12 ve 6 sezon uzaklıkta olduğu için `season_carry` ile sönümlenmiş
halde taşınıyordu; hiçbir donmuş AO First parametresi bu sezonlardan türemez.
Doğru çözüm beklenen sayıyı resmî fikstür kaydından pinlemektir ve holdout
kilidinden sonraki bakım turuna bırakılmıştır.

Yukarıdaki `81 kulüp` sayacı hem tekrar temizliğini hem bu iki sezonun
düşmesini birlikte içerir; iki bileşen ayrıştırılmamıştır.

Kapı kararı kaynak-bağımlıdır ve bu iki sezonun düşmesi "daha eksik oldukları"
anlamına gelmez. GEO 2020 secondary'nin kendi ölçüsüne göre (`94/184 = 0.511`)
GEO 2014'ten (`0.652`) daha kötüdür, ama primary fallback'i (`92/72 = 1.278`)
sayesinde production'a girer. Kalite dosyasında **22 ülke** aynı lig için birden
fazla `table_expected_matches` taşır; `ACCEPTED` bir satır sezonun tam olduğunu
kanıtlamaz. Ayrıntı `docs/ai/DATA_CONTRACTS.md` §11.2'de.

Boşluk listesi `tests/test_domestic_ucl_uel_expansion.py` içindeki üç regresyon
testiyle pinlenmiştir: kalite audit'inde iki sezonun boş olması, production
girdisinde bulunmamaları ve ülke bazında başka iç sezon boşluğu doğmaması.
Liste büyürse test kırılır.

## Değişmeyenler ve kanıt sınırı

AO First, rating config, exposure cap, katılım k'sı, kupa ağırlığı, bilinmeyen
sıra tabanı, served blend ve production Poisson transfer katsayıları **birebir
korundu**. Aktif Structural Logistic katsayıları ve intercept'leri de öncekiyle
aynı. Artifact dosyasının hash'i metadata/provenance değişince değişebilir.
Config ID yine `9c2ed2fad929252b`.

AO çekirdeğinin `model_summary`, `fold_summary`, `competition_summary` ve
`dependency_uncertainty` CSV'leri önceki snapshot ile byte-identical.
Bu yüzden AO First'in üstündeki statik ablation/kalibrasyon zinciri tekrar
seçilmedi. ML `--blend-weight 0.9`, ensemble `--prospective-poisson-weight 0.5`
ile çalıştırıldı. Eski Poisson yüzeyi kullanılmadı: 54 aday önce bağımsız
yeniden hesaplandı; sonraki backtest bu yeni, hash'i kaydedilmiş yüzeyi okudu.
[Yüzey provenance](domestic_integrity_fix_2026_08_28/domestic_surface_provenance.json).

Artifact builder'ın kanıt metrikleri artık eski literal sayılardan değil,
yeniden üretilen ensemble kararından alınır. İki contract, manifest, state,
input hash'leri ve current evaluation aynı zincire bağlandı.

## Güncellenen tarihsel ölçüm

Pencere **2018/19–2025/26**, 6.340 geliştirme maçı; **2020/21–2025/26**
altı outer fold, **4.884** değerlendirme maçı. Aşağıdaki ensemble fold bazında
seçim yapan **nested tarihsel kol**; sabit production `%50/%50, rho=0`
politikasının birebir tam-pencere ölçümü değildir.

| Kol | Brier | Log-loss | Accuracy |
| --- | ---: | ---: | ---: |
| Current AO | 0.566413 | 0.956259 | 0.559173 |
| Current ML | 0.562750 | 0.951215 | 0.560197 |
| Nested ensemble | 0.562065 | 0.949965 | 0.561220 |

Current ML'ye karşı fold kazanımı Brier **4/6**, log-loss **5/6**.
Fold ağırlıkları `0.6/0.9/0.7/0.3/0.1/0.2`; prospective pin hâlâ `0.5`.
Pin olmasa bu yüzey `0.4` seçerdi; production'a taşınmadı.

| Baseline | Brier farkı [%95 CI] | Log-loss farkı [%95 CI] |
| --- | --- | --- |
| Current AO | -0.004348 [-0.006442, -0.002186] | -0.006294 [-0.009636, -0.002765] |
| Current ML | -0.000685 [-0.001662, +0.000218] | -0.001249 [-0.002901, +0.000265] |

4.000 bootstrap, conservative envelope. [Foldlar](production_prediction/fold_results.csv),
[CI](production_prediction/dependency_uncertainty.csv), [segmentler](production_prediction/competition_coverage_summary.csv).
Current ML'ye karşı UCL/UEL loss yönleri olumlu; UECL'de Brier/log-loss
`+0.000628/+0.000780`, NONE coverage'da `+0.000441/+0.000382`.
Otomatik ensemble kararı yine `KEEP_SHADOW`; mevcut operasyonel karar değişmedi.
Genişletilmiş Poisson araştırma kolunun eski ve yeni kararı `REJECT`; bu onarım
yeni bir araştırma terfisi olarak sunulmaz.

Sabit served politikanın ayrı ClubElo benchmark'ında **363 eşleşmiş maç**:
Brier **0.579166**, log-loss **0.973577**. ClubElo'ya Brier farkı
`+0.004183`, %95 CI `[-0.023280,+0.033869]`; log-loss farkı `+0.006758`,
CI `[-0.032070,+0.048094]`. Bu küçük örneklem tam-pencere veya prospective kanıt değildir.

## Doğrulama

- `python3 -m pytest -q`: **1189 passed**, 1 fiziksel CPU sayımı uyarısı,
  **146.45 saniye**. Bu tur 14 regression testi eklendi.
- Current safety/data audit: **23/23**.
- `python3 -m ruff check src/`: **27 mevcut bulgu**; temiz değildir.
  Değiştirilen dört kaynak dosyasında önceki bazla karşılaştırmada yeni bulgu **0**.
- Bağımsız replay checkpoint ile tam eşit; eski girdi merge/replay'de reddediliyor.
- Contract/manifest/artifact/input/snapshot zinciri: **9/9** kontrol.
- 2026/27 feature replay: **86 satır**, 82 ACTIVE_ENSEMBLE, 4 fallback
  (**%4.65**); `rows_with_imputed_model_input=2` ayrı sayaçtır. Retrospective
  üretimdir; geçmişte kickoff öncesi ledger yazıldığına kanıt değildir.
- Dört PDF yeniden üretildi: 32 sayfa görsel yerleşim kontrolü; sayfa dışına
  taşan metin yok. Değişen tablolar ayrıca tam çözünürlükte incelendi.
- `git diff --check` geçti. Önceki kullanıcı/Tur 1 değişiklikleri geri alınmadı;
  commit veya push yapılmadı.

[Zincir hash kanıtı](domestic_integrity_fix_2026_08_28/final_chain_verification.json).
Production state SHA-256:
`0b82dc133e9f67f5459fd42eca969560b786a872e6ca37d644010f1e96d333d1`.
Komut logları, bağımsız replay scripti ve önceki tam veri/artifact kopyaları
`output/domestic_integrity_fix_2026_08_28/` altında saklanır.

## Açık kalan kapsam

**#3 kapanmadı:** `ml_backtest.py` içindeki hardcoded `2026/27_UNTOUCHED`
gerçek bir tarih/veri guard'ı değildir. **2026-09-08 kilidinden önce ayrı
olarak kapatılmalıdır**; bugünkü checkpoint bozulmasının nedeni olarak sayılmaz.
**#2 ve #6–#11**, kullanıcının kararı doğrultusunda ertelenmiştir.
Özellikle UTC fold sınırı bulgusu giderilmedi; bu tarihsel metrikler leakage-free
veya untouched holdout sertifikası değildir. Mod/medyan coverage eşiği de
resmî lig formatı yerine geçmez.

Prospective append-only prediction ledger ve LIVE_DATA_INGESTION §18 servis
kabulü bu turda kurulmadı/doğrulanmadı. **İstenen veri/state kusurları kapatıldı;
bu rapor tek başına canlıya alınabilirlik onayı değildir.**
