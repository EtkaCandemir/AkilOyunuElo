# UCL/UEL Domestic Poisson Candidate Expansion

Bu klasor, UCL/UEL oncelikli 2026/27 takim evreni icin yerel lig verisi
genisletme girdisidir. Onayli kaynak ve kimlik denetiminden gecen candidate
veri, `2026-08-21-ucl-uel-domestic-poisson-coverage` revizyonunda production
Domestic Poisson checkpoint'ine alinmistir.

## Kapsam

- Dondurulmus hedef evreni: 80 mevcut UCL/UEL lig asamasi ya da play-off takimi.
- Yeni ligler: `ALB, ARM, AZB, BUL, CRO, GEO, HUN, KAZ, LIT, ROM, SCO, SLO,
  SRB, SVK, UKR`.
- Tarihsel pencere: 2013/14-2025/26.
- Candidate state uygunluk kurali: en az iki kabul edilmis sezon ve 40 yerel
  mac; aksi durumda servis AO olasilik fallback'ine doner.

## Kaynak Sozlesmesi

1. TheSportsDB Premium birincil kaynaktir. UTC, skor, benzersiz event ve final
   tabloya gore en az %95 kapsama kapisini gecen sezonlar kabul edilir.
2. Birincil sezon reddedilirse, ayni lig-sezonun tamami Hugging Face'teki
   API-Football fixture arsivinden alinabilir. Bu arsivde tablo endpoint'i
   olmadigi icin %95 kapisi ligin modal tamamlanmis sezon fixture sayisina gore
   uygulanir.
3. Bir lig-sezonda satir bazinda iki kaynak karistirilmaz. Birincil kabul edilirse
   ikincil satirlar kullanilmaz.
4. Eksik xG, tahmin edilmis skor veya gelecek mac sonucu bu veri setine girmez.

## Dosyalar

- `target_team_audit.csv`: dondurulmus 80 takimin kimligi ve secim nedeni.
- `target_coverage_audit.csv`: baseline/candidate kapsama ve acik fallback.
- `league_registry.csv`: hedef lig katalog kimlikleri.
- `league_season_quality.csv`: sezon bazli kaynak ve kalite karari.
- `expansion_matches_primary.csv`, `expansion_matches_secondary.csv`:
  ham-normalize aday kaynaklar.
- `expansion_matches_selected.csv`: tek-kaynak kuralindan sonra secilen
  genisleme maclari.
- `domestic_matches_candidate.csv` ve `domestic_team_bridge_candidate.csv`:
  Poisson backtesti/state'i icin candidate girdileri.
- `live_2026_27_matches.csv`: belirtilen cutoff'tan once tamamlanan, causal
  2026/27 yerel maclar.
- `source_manifest.json`: kaynak snapshot ve SHA-256 denetimi.

## Causal State

`live_2026_27_matches.csv` yalniz `domestic_kickoff < european_kickoff`
kosulunda bir Avrupa tahminine girebilir. Ayni kickoff batch'indeki yerel
sonuclar sonraki tahmine sizmaz. Kaynak degisen liglerde, yalniz dogrulanmis AO
kulup kimlikleri icin internal state anahtari kanoniklestirilir; provider event
kimlikleri ise denetim icin korunur.

Bu klasorun tarihsel aday degerlendirmesi:
`reports/domestic_poisson_ucl_uel_expansion/evaluation_report.md`.

Production checkpoint, `artifacts/production_prediction/` altindaki
`domestic_poisson_state_2026_27.json` dosyasidir. Model parametreleri bu
genislemede degismemistir; yalniz kimlik ve yerel-gecmis kapsami genislemistir.
