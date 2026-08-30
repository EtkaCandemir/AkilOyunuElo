# 2026/27 Prediction Ledger Provenance

## Revision 0

İlk 396 prediction kaydı silinmedi veya yeniden yazılmadı. Bu kayıtlarda
`recorded_at_utc = 2026-08-30T09:59:21.012061+00:00`, fakat
`generated_at_utc = 2026-09-08T10:45:00+00:00` idi. Dolayısıyla tahminler
kickoff'tan önce görünse de `recorded_at_utc < generated_at_utc` olduğu için
zaman semantiği fiziksel olarak tutarsızdı. Revision 0, holdout sonucu için
kilitli tahmin sayılmaz.

Bu satırlar ledger'ın append-only sözleşmesi nedeniyle yerinde bırakıldı.
İlk 396 satırın SHA-256 değeri değişmedi:
`12e333ababf6c837a9f0cb72561244257c12236780c5c7295a77944742c921fb`.
Bu zincirin eski head hash'i:
`bc09379e45df2a8ff2462bf1dbe7358ef47e38f7df6b856274c5ea6d145f7985`.

## Revision 1

396 fikstür 30 Ağustos 2026'da gerçek üretim saatiyle yeniden hesaplandı ve
mevcut zincire revision 1 olarak eklendi:

- `generated_at_utc = 2026-08-30T13:01:03.262137+00:00`
- `recorded_at_utc = 2026-08-30T13:01:39.138475+00:00`
- en erken kickoff: `2026-09-08T16:45:00+00:00`
- zaman kuralı: `generated_at_utc <= recorded_at_utc < kickoff_utc` — 396/396

Revision 1, `2026-08-28-domestic-provider-season-and-fixture-integrity-fixes`
revision'ının ürettiği tahmindir. **Artık kilitli tahmin değildir**: aşağıdaki
model değişikliğiyle geçersiz kaldı. Protokol uyarınca kickoff'tan önce açıkça kaydedilmiş daha sonraki bir
operasyonel revision oluşursa yalnız o maç için en yüksek geçerli revision kilitli
tahmin olur; eski revision'lar yine zincirde kalır.

Revision 1 append'inden sonraki head hash:
`a0debe686ac202ca6103044b5f6c95a629fd54d9960751fb69022357fbcc3dbc`.
Güncel anchor: `ledger_anchor_2026_27.json`.

## Revision 2

`european_tail_beta` `0` iken Avrupa geçmişi normu `1.0`'da kesiliyordu ve
benchmark'ı aşan bütün kulüpler tek bir European Prior alıyordu — 2026/27'de 14
kulüp. Kesme kaldırıldı (`beta = 1`), yani AO First değişti ve revision 1'in
tahminleri farklı bir modelden gelmiş oldu.

- production revision: `2026-08-30-european-prior-tail-no-truncation`
- `generated_at_utc = 2026-08-30T17:48:12.234191+00:00`
- `recorded_at_utc = 2026-08-30T17:48:48.971512+00:00`
- en erken kickoff: `2026-09-08T16:45:00+00:00`
- zaman kuralı `generated_at_utc <= recorded_at_utc < kickoff_utc` — 396/396

Servis edilen 1X2, revision 1'e göre en çok `11.19` puan kaydı; UCL'de ortalama
`2.95`, UEL'de `0.25`, UECL'de `0.17` puan. 396 satırın tamamı değişti çünkü ML
modeli yeni AO özellikleriyle yeniden eğitildi.

**Revision 2, bu 396 fikstür için holdout başlangıcında kilitli ve
değerlendirilecek tahmindir.** Revision 0 ve 1 zincirde kalır; ilk 792 satırın
SHA-256 değeri değişmedi:
`164052913feeb01f71da3469b537cb098f91f30d1bb486a52fdf0cae0a25c785`.

Revision 2 append'inden sonraki head hash:
`11d45a81c12ac0586272f9f22b2f329cb0b8ffc9c44fdb2920e098e6ec90848a`.
