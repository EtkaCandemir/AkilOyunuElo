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

Revision 1, bu 396 fikstür için holdout başlangıcında kilitli ve değerlendirilecek
tahmindir. Protokol uyarınca kickoff'tan önce açıkça kaydedilmiş daha sonraki bir
operasyonel revision oluşursa yalnız o maç için en yüksek geçerli revision kilitli
tahmin olur; eski revision'lar yine zincirde kalır.

Revision 1 append'inden sonraki head hash:
`a0debe686ac202ca6103044b5f6c95a629fd54d9960751fb69022357fbcc3dbc`.
Güncel anchor: `ledger_anchor_2026_27.json`.
