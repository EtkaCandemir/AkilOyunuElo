# AO European Elo - Agent Instructions

Bu dosya, bu repository ile calisan tum coding agent'lar icin baglayici giris
sozlesmesidir. Ayrintili proje baglami icin once [CODEX.md](CODEX.md) okunmalidir.

## Zorunlu Okuma Sirasi

1. `CODEX.md`
2. `contracts/ao_european_elo_v2_production.json`
3. Ilgili `src/ao_elo/*.py` kaynak dosyalari
4. `reports/current_model/active_model_snapshot.json`
5. `reports/current_model/current_model_evaluation_report.md`

Birbiriyle celisen bilgi varsa bu oncelik sirasi kullanilir. Eski PDF, README,
rapor veya sohbet metni production parametresi icin otorite degildir.

## Modeli Degistirme Kurallari

- Kullanici acikca production aktivasyonu istemedikce production contract'i,
  `AOEuropeanEloConfig.active()` veya `DynamicEloConfig.calibrated_v2()`
  degistirilmez.
- Research scriptinin var olmasi feature'in aktif oldugu anlamina gelmez.
- Her yeni katman once mevcut production'a karsi incremental ablation olarak
  test edilir.
- Parametre secimi test sezonundan yapilmaz. Expanding/walk-forward ayrimi ve
  exact-UTC kronoloji korunur.
- `2026/27` prospective ledger verisi gelmeden untouched holdout iddiasi
  yapilmaz.
- AO First Elo, Power Elo, Achievement Reserve, Progression Bonus ve AO Live
  Elo birbirine karistirilmaz.

## Degismez Invariantlar

- Match Power Elo guncellemesi sifir-toplamlidir.
- `AO Live Elo = Power Elo + Achievement Reserve + Progression Bonus`.
- Aktif Achievement Reserve sifirdir; aktif Progression Bonus sifir-toplamli
  degildir ve sezon sonunda sifirlanir.
- Penalty shootout golleri field score'a eklenmez.
- Shootout, esit olmayan 90/120 field score sonucunu beraberlik yapmaz; yalniz
  gol farki ve xG ek sinyallerini kapatir.
- Ayni kickoff UTC batch'indeki tahminler, batch sonucu islenmeden kilitlenir.
- Eksik Avrupa gecmisi sifir varsayilmaz; acik all-zero kulup satiri gerekir.
- xG yalniz iki takim icin birlikte ve uygun zaman kapsamiyla varsa uygulanir.
- Ratingler `500-2000` araliginda kirpilmaz; bu yalniz referans bandidir.
- Aktif served 1X2, `%50 Current ML + %50 AO Domestic Poisson (rho=0)` log
  blend'idir; AO Live Elo'ya geri beslenmez ve hata halinde Current AO 1X2'ye
  doner.

## Kod ve Test Disiplini

- Manuel duzenlemeler `apply_patch` ile yapilir.
- Kullaniciya ait mevcut degisiklikler geri alinmaz.
- Dar degisiklikte ilgili testler, model/contract degisikliginde tam `pytest -q`
  calistirilir.
- Uretilen `output/` dosyalari kaynak kod sayilmaz; tekrar uretilebilir rapor
  artefact'laridir.
- Model davranisi degisirse contract, test, `CODEX.md` ve ilgili `docs/ai/`
  belgesi ayni degisiklikte guncellenir.

## Dil ve Raporlama

Kod ve kolon adlari Ingilizce, model aciklamalari tercihen Turkce yazilir.
Sonuc raporunda yalniz "testler gecti" denmez; kullanilan veri penceresi, mac
sayisi, baseline, pooled/fold metrikleri, segmentler ve guven araliklari verilir.
