# Gol Farkı Shadow Parametre Araması

## Kapsam

- Sezonlar: `2018/19-2025/26`.
- Formül değişmedi; `13 x 9` alpha/tau yüzeyi, alpha=0 tekrarları tekilleştirilerek test edildi.
- Scale, H, K, AO First Elo ve carry değiştirilmedi.
- Fixed yüzey post-hoc duyarlılık, nested seçim ise dürüst unseen kanıttır.

## Geniş Yüzey Sonucu

- En düşük fixed OOS Brier: `alpha=0.3`, `tau=800`, ΔBrier `-0.000605`.
- Ranking-first shadow adayı: `alpha=0.125`, `tau=800`.
- En iyi Brier'in `0.00005` çevresindeki plateau aday sayısı: `3`.

## Nested Walk-forward

- ΔBrier: `-0.000309`.
- Δlog-loss: `-0.000457`.
- Brier zarfı: `[-0.000541, -0.000095]`.
- Log-loss zarfı: `[-0.000799, -0.000139]`.
- Nested sıralama: tam sıfır toleransla `2/5`, ileriye dönük pratik eşiklerle `5/5` fold güvenli.

| Fold | Test | Seçilen alpha | Seçilen tau |
| ---: | --- | ---: | ---: |
| 1 | 2020/21 | 0.25 | 800 |
| 2 | 2021/22 | 0.225 | 650 |
| 3 | 2022/23 | 0.225 | 650 |
| 4 | 2023/24 | 0 | 300 |
| 5 | 2024/25 | 0 | 300 |
| 6 | 2025/26 | 0 | 300 |

## Prospective Shadow Kolları

| Kol | alpha | tau | ΔBrier | Δlog-loss |
| --- | ---: | ---: | ---: | ---: |
| BASE | 0 | 300 | +0.000000 | +0.000000 |
| PRE_SPECIFIED | 0.1 | 300 | -0.000174 | -0.000250 |
| PRIOR_GRID_BEST | 0.2 | 400 | -0.000366 | -0.000527 |
| EXTENDED_BEST | 0.125 | 800 | -0.000296 | -0.000426 |

## Önceden Belirlenen Aday

- `alpha=0.10, tau=300` Brier'da `6/6` fold kazandı; toplam ΔBrier `-0.000174`, Δlog-loss `-0.000250`.
- Dependency zarfı Brier için `[-0.000288, -0.000050]`, log-loss için `[-0.000410, -0.000077]`.
- Pooled sıralama değişimi Spearman `+0.000626`, pairwise `+0.000094`.
- Tam sıfır toleransla `2/5`, pratik eşiklerle `5/5` fold güvenli.
- Maksimum gol çarpanı `1.138347`, yüzde 99 `1.108772`; maksimum mutlak Power delta `93.018`.
- Pratik eşikler bu tarihsel sonuç görüldükten sonra tanımlandığı için geçmiş veride geriye dönük terfi kanıtı sayılamaz; 2026/27 shadow ölçümünde önceden kayıtlı kapı olarak kullanılacaktır.

## Karar

- Gol farkı ana kod tabanının kalıcı bir bileşenidir, fakat aktif rating etkisi prospective kapılar geçilene kadar sıfırdır.
- PRE_SPECIFIED, PRIOR_GRID_BEST ve gerekirse EXTENDED_BEST aynı maçlarda paralel shadow state olarak izlenecektir.
- En az 300 prospective maç ve 75 UCL maçı oluşmadan aktivasyon kararı verilmeyecektir.
- Forward sıralama kapısı: pooled fark negatif olmayacak; tek fold Spearman gerilemesi `0.0050`, pairwise gerilemesi `0.0025` değerini aşmayacak.
- Bu ara kapı adayı aday gösterebilir; tam aktivasyon bir sonraki sezonun forward doğrulamasını ve manuel model kararı gerektirir.
- Brier/log-loss yönü, dependency zararı ve forward sıralama birlikte değerlendirilecektir; yalnızca modelin daha karmaşık görünmesi terfi gerekçesi değildir.

Makinece okunabilir karar: `SHADOW_EVIDENCE_COLLECTION`.
