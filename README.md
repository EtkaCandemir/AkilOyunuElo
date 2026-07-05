# AO European Elo

AO European Elo, UEFA kulüp turnuvaları için sezon öncesi başlangıç rating'i üreten bir Python/Pandas hesap motorudur. Model; ülke/lig gücü, yerel lig/kupa başarısı ve kulübün son yıllardaki Avrupa performansını tek bir başlangıç puanında birleştirir.

Bu proje şu an pre-calibration aşamasındadır. Benchmark değerleri tarihsel veriyle kalibre edilmeden production modeli olarak kullanılmamalıdır.

## Ana Formül

```text
AO First Elo =
Domestic Prior
+ European Exposure * (European Prior - Domestic Prior)
```

- `Domestic Prior`: Ülke/lig gücü ve yerel başarı sinyali.
- `European Prior`: Kulübün son 5 sezondaki Avrupa performansı.
- `European Exposure`: Avrupa performansı sinyalinin kaç sezon/maçlık veriye dayandığı.

Avrupa geçmişi olmayan takımlar cezalandırılmaz. Bu durumda `European Exposure = 0` olur ve final puan `Domestic Prior` değerine eşit kalır.

## Proje Yapısı

```text
src/ao_elo/
  config.py        # Model parametreleri
  features.py      # Feature hesapları
  scoring.py       # Rating formülleri
  validators.py    # Input validation kuralları
  pipeline.py      # CSV/dataframe hesap pipeline'ı

tests/
  test_achievement.py
  test_exposure.py
  test_scoring.py

data/pilot_10_teams/
  teams.csv
  country_coefficients.csv
  domestic_context.csv
  club_european_points.csv

scripts/
  run_pilot_10_teams.py
  build_pilot_inspection_report.py
  build_data_requirements_pdf.py

output/
  pilot_10_teams/
  pdf/
```

## Gerekli CSV Dosyaları

Model dört ana input CSV bekler:

1. `teams.csv`
   - Takım kimliği, ülke, ülke kodu ve yerel lig bilgisi.

2. `country_coefficients.csv`
   - Son 5 sezon UEFA ülke katsayı puanları.
   - Lig/ülke gücü hesabında kullanılır.

3. `domestic_context.csv`
   - Lig pozisyonu, ligdeki takım sayısı, lig şampiyonluğu, kupa şampiyonluğu ve Avrupa giriş bilgisi.
   - Yerel başarı skorunu üretir.

4. `club_european_points.csv`
   - Kulübün son 5 sezon Avrupa puanları, Avrupa oynayıp oynamadığı, maç sayıları ve match cap değerleri.
   - European Prior ve European Exposure hesabında kullanılır.

Detaylı alan sözlüğü için:

```text
output/pdf/AkilOyunu_VeriAnlamlandirma.pdf
```

## Kurulum

```bash
python3 -m pip install -r requirements.txt
```

## Testleri Çalıştırma

```bash
pytest -q
```

Beklenen mevcut sonuç:

```text
9 passed
```

## Pilot Veri Setini Çalıştırma

10 takımlı sentetik pilot veri seti için:

```bash
python3 scripts/run_pilot_10_teams.py
```

Bu script şunu üretir:

```text
output/pilot_10_teams/ao_first_elo_pilot_output.csv
```

Pilot inspection raporu için:

```bash
python3 scripts/build_pilot_inspection_report.py
```

Üretilen dosyalar:

```text
output/pilot_10_teams/pilot_inspection_report.md
output/pilot_10_teams/pilot_inspection_table.csv
```

## PDF Dokümanları

Projede üç açıklayıcı PDF bulunur:

```text
AkilOyunu_Elo_Model_Aciklayici.pdf
AkilOyunu_Elo_Model_Kisa.pdf
output/pdf/AkilOyunu_VeriAnlamlandirma.pdf
```

- İlk PDF detaylı model spesifikasyonudur.
- İkinci PDF kısa model anlatımıdır.
- Üçüncü PDF veri ihtiyaçları ve alan sözlüğüdür.

## Kalibrasyon Notu

Model iki dış benchmark parametresi ister:

```text
Country_Strength_Benchmark
European_History_Benchmark
```

Pilot testte bu değerler sabit test parametresi olarak kullanılmıştır. Gerçek kullanımda son 8-10 sezonluk tarihsel dağılım ve backtest ile kalibre edilmelidir.

## Durum

Bu repo şu an:

- Modüler Python hesap motoru içerir.
- Input validation kurallarını uygular.
- Sentetik pilot veri seti ve pilot raporu içerir.
- Unit testlerle temel model davranışlarını doğrular.

Maç sonrası Elo güncelleme modeli bu sürümün kapsamında değildir.
