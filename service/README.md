# AO Elo Service (FastAPI sarmalayıcı)

`src/ao_elo/*` içindeki model kodunu **hiç değiştirmeden** HTTP üzerinden erişilebilir kılan ince bir katman. akiloyunu-api (Laravel) bu servise sadece hesap sonucu almak için istek atar; tüm kalıcılık/state Laravel tarafındadır — bu servis hiçbir şey diske/hafızaya yazmaz.

## Çalıştırma

```bash
pip install -r requirements.txt
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

`GET /health` → aktif config kimlikleri + production ML/Poisson ensemble'ının yüklenip yüklenemediğini (ve neden) raporlar.

## Endpoint'ler

- `POST /bootstrap-season` — 4 sezon-başı tabloyu (`teams`, `country_coefficients`, `domestic_context`, `club_european_points`, DATA_CONTRACTS.md §2-5 şemasıyla) alır, `pipeline.compute_ao_first_elo()` ile AO First Elo üretir.
- `POST /predict` — bir fikstür + her iki takımın güncel rating bileşenlerini (`power_elo`, `achievement_reserve`, `progression_bonus_ucl/uel/uecl`) alır. Önce `dynamic.lock_prediction()` ile ham AO 1X2'yi hesaplar; `contracts/ao_european_elo_v2_production.json` + `artifacts/production_prediction/` mevcutsa `ProductionPredictionService` ile ML+Poisson blend'ini dener. Gerekli ML feature'ları (rolling Avrupa formu) sağlanmadığı sürece **her zaman** `FALLBACK_CURRENT_AO` ile dürüstçe ham AO olasılıklarına düşer — bu doğru ve beklenen davranış (`docs/ai/LIVE_DATA_INGESTION.md` §6: lisanslı/tam sağlayıcı olmadan resmi fallback korunur).
- `POST /settle` — aynı fikstür + sonuç (skor, penaltı, opsiyonel xG) alır, `dynamic.update_match()` ile zero-sum Power güncellemesini VE (eligible stage'de tie-decider ise) UCL/UEL/UECL progression bonusunu **otomatik olarak birlikte** hesaplar.

## Bilinen sınırlamalar (dürüstçe not düşüldü)

- Her istek, gönderilen iki takımın rating'inden **geçici** bir `SeasonState` kurar (Laravel state'i kalıcı tutuyor, bu servis tutmuyor). `qualifier_to_main_carry` aktif configde `1.0` (no-op) olduğu için bunun hesap sonucuna etkisi yoktur — bu değer değişirse bu servisin de güncellenmesi gerekir.
- `/predict`, tam ML+Poisson production ensemble'ını çalıştırmak için gereken rolling Avrupa form feature'larını (bkz. `scripts/build_2026_27_prediction_features.py`) şu an **almıyor** — bu yüzden gerçek ML artifact'ları yüklense bile her satır muhtemelen `FEATURE_OR_STATE_INVALID` ile fallback'e düşecektir. Bu, Faz 0'ın kasıtlı olarak dar tutulan kapsamıdır; feature köprüsü ayrı bir iş.
- akiloyunu-api tarafının şu anki `/predict` ve `/settle` payload'ı tek bir toplu `home_rating`/`away_rating` gönderiyor — bu servis ise ayrıştırılmış (`power_elo`, `achievement_reserve`, `progression_bonus_*`) bir yapı bekliyor. Laravel tarafının bu servisle gerçekten konuşabilmesi için payload'ını güncellemesi gerekecek (akiloyunu-api'nin kendi `ao_rating_state` tablosu bu bileşenleri zaten ayrı ayrı tutuyor, sadece gönderilmiyor).
- Laravel tarafındaki `AoProgressionReconciliationService` (kendi başına FORMULAS.md §17'yi hesaplıyor) bu servisin `/settle` yanıtıyla ÇAKIŞABİLİR — `update_match()` progresyonu zaten otomatik hesaplıyor. Gerçek entegrasyonda muhtemelen Laravel'in kendi hesaplaması yerine bu servisin döndürdüğü değerler kullanılmalı.
- `prediction_status` fallback değeri tam olarak `"FALLBACK_CURRENT_AO"`dur (Laravel tarafında bir yerde `"CURRENT_AO_1X2"` varsayılmıştı — bu düzeltilmeli).

## Test

Gerçek çağrılarla (fixture + sonuç) uçtan uca doğrulandı: `/health`, `/bootstrap-season` (tek takım, tam AO First Elo çıktısı), `/predict` (LEAGUE_STAGE + gelecekteki Q1/LAST_16 fikstürleri, doğru fallback), `/settle` (Q1'de progression YOK — doğru, LAST_16 tie-decider'da progression VAR ve tam `12.0` — FORMULAS.md §17 ile birebir eşleşti, zero-sum Power güncellemesi doğrulandı).

Yerel test için (Python 3.10'da `scipy==1.17.1`/`scikit-learn==1.8.0` pinleri 3.11+ gerektirdiğinden, `requirements.txt`'teki pinler yerine sürüm belirtmeden kurulabilir — sadece `service/app.py`'yi ve core `pipeline`/`dynamic` modüllerini test etmek için yeterlidir, gerçek ML artifact'larıyla bit-birebir eşleşmeyebilir):

```bash
python -m venv .venv
.venv/Scripts/pip install pandas numpy joblib fastapi uvicorn pydantic scipy scikit-learn httpx pytest
PYTHONPATH=. .venv/Scripts/python -c "from fastapi.testclient import TestClient; from service.app import app; print(TestClient(app).get('/health').json())"
```
