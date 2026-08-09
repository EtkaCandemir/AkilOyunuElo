# AO xG Backtest Dataset

Bu klasör, AO European Elo'nun gol farkı ve xG ablation çalışması için
denetlenmiş matched-sample verisini içerir.

## Kaynak ve Lisans

- Kaynak: `eatpizzanot/soccer-dataset`
- Kaynak URL: https://huggingface.co/datasets/eatpizzanot/soccer-dataset
- Veri sözlüğü: https://huggingface.co/datasets/eatpizzanot/soccer-dataset/blob/main/data_dictionary.md
- Alt kaynaklar: API-Football ve football-data.co.uk
- Lisans: CC-BY-4.0
- Metrik: `coarse_zone_derived_xg`

Kaynağın kendi veri sözlüğüne göre bu değer Opta/StatsBomb tipi per-shot xG
değildir; şut bölgelerinden türetilmiş kaba bir sağlayıcı tahminidir. Bu nedenle
sonuçlar yalnızca exploratory/shadow kanıt sayılır ve production aktivasyonu için
tek başına yeterli değildir.

## Sözleşme

- Eksik xG sıfırla doldurulmaz.
- Yalnızca 90 dakikada `FT` biten, iki taraf xG'si sonlu ve negatif olmayan maçlar
  ablation örneklemine alınır.
- Uzatma ve penaltı maçları, sağlayıcının süre kapsamı denetlenemediği için dışarıda
  bırakılır.
- Eşleşme anahtarı turnuva, UTC tarih, skor ve yüksek güvenli takım adı eşleşmesidir.
- xG yalnızca maç bittikten sonra rating güncellemesinde kullanılır.

## Satır Sayıları

- AO maçları: 6340
- Kaynakta iki taraf xG bulunan maç: 787
- AO ile eşleşen xG maçı: 426
- Ablation için uygun maç: 417

## Uygun Kapsam

| season | competition | eligible_xg | ao_matches |
|---|---:|---:|---:|
| 2019/20 | UCL | 9 | 210 |
| 2019/20 | UEL | 24 | 511 |
| 2020/21 | UCL | 60 | 178 |
| 2020/21 | UEL | 104 | 362 |
| 2024/25 | UCL | 40 | 279 |
| 2025/26 | UCL | 111 | 281 |
| 2025/26 | UECL | 34 | 409 |
| 2025/26 | UEL | 35 | 271 |

`source_manifest.json` kaynak hash'lerini ve production engelini,
`identity_audit.csv` her AO maçının eşleşme kararını,
`coverage_matrix.csv` sezon-turnuva kapsamını içerir.
