# AO 2025/26 UEFA Maç ve xG Veri Seti

Bu klasör 2025/26 sezonundaki UCL, UEL ve UECL maçlarını AO takım
kimlikleriyle birleştirir. Ana CSV her maç için tek satır içerir. AO First Elo ve
AO Live Elo bu çalışma sırasında değiştirilmemiştir.

## Kaynak Sözleşmesi

- Maç kimliği, tarih ve saha skoru UEFA `match.uefa.com/v5/matches` kaynağıyla
  doğrulanır.
- Birincil xG, FotMob'un herkese açık maç sayfasını besleyen kimlik doğrulamasız
  sayfa-verisi yanıtında açıkça gösterilen `Expected goals (xG)` toplamıdır.
- FotMob xG şut kalitesi temellidir. Maç içi penaltılar dahildir; penaltı atışları
  toplamdan çıkarılır. Uzatma oynandıysa yalnız 120 dakikayı kapsadığı
  doğrulanabilen değer analize uygun sayılır.
- Eksik veya kapsamı belirsiz xG sıfırla, ortalamayla ya da tahminle doldurulmaz.
- API-Football tabanlı coarse xG yalnız `secondary_xg_*` kolonlarında tutulur ve
  birincil FotMob kolonlarına taşınmaz.
- FotMob açıklaması: https://www.fotmob.com/pt-BR/faq

Kamuya açık erişim, otomatik olarak yeniden dağıtım lisansı vermez. Bu CSV veya
FotMob kaynaklı alanlar herkese açık bir repoda yayımlanmadan önce güncel kaynak
kullanım ve yeniden dağıtım koşulları ayrıca kontrol edilmelidir.

## Kapsam

| Turnuva | Maç | FotMob xG | Analize uygun | Kapsama |
|---|---:|---:|---:|---:|
| UCL | 1384 | 1032 | 1032 | 74.6% |
| UECL | 2073 | 717 | 717 | 34.6% |
| UEL | 1427 | 1078 | 1078 | 75.5% |

- Toplam maç: 4884
- Benzersiz AO takımı: 237
- İki kaynakta ortak xG maçı: 182
- Çalıştırma modu: offline/cache

## Dosyalar

- `uefa_2025_26_matches_with_xg.csv`: 961 maçlık ana analiz tablosu.
- `xg_identity_audit.csv`: UEFA yenileme, FotMob kimlik ve xG kabul denetimi.
- `xg_coverage_summary.csv`: turnuva ve tur bazlı kapsam.
- `xg_source_comparison.csv`: iki xG kaynağının ortak maçları.
- `source_manifest.json`: kaynak URL'leri, SHA-256 değerleri ve sözleşme.

Ham yanıtlar `_source_cache/` altında tutulur ve Git'e eklenmez. Aynı cache ile
yeniden çalıştırıldığında ana CSV deterministik olarak üretilir.
