# AO European Elo v2 Pilot Sonuclari

Model: `ao-european-elo-v2.0-dev-freeze`

V2, v1.1'in 500 etrafındaki affine ölçek dönüşümüdür; bu nedenle pilot takım sırası birebir korunur. 500-2000 bir referans bandıdır, hard cap değildir.

## Statik özet

| pilot | teams | minimum | median | maximum | validation_warning_rows | v1_v2_rank_identical |
| --- | --- | --- | --- | --- | --- | --- |
| synthetic | 10 | 841.5141430601294 | 1356.925827714495 | 1914.7683621166716 | 0 | True |
| real | 10 | 1398.414730744317 | 1749.0431231975917 | 1992.8698752228167 | 0 | True |

## Gerçek pilot sıralaması

| ao_first_elo_rank | team_name | ao_first_elo |
| --- | --- | --- |
| 1 | Arsenal | 1992.8698752228167 |
| 2 | Sporting CP | 1926.710636167372 |
| 3 | Benfica | 1881.3345310017787 |
| 4 | Shakhtar Donetsk | 1764.6842322861187 |
| 5 | Galatasaray | 1756.281979019065 |
| 6 | AZ Alkmaar | 1741.8042673761188 |
| 7 | Slavia Praha | 1621.8559129513487 |
| 8 | Pafos | 1444.0355195824964 |
| 9 | Como | 1421.4363341035685 |
| 10 | Omonia Nicosia | 1398.414730744317 |

## Dinamik smoke test

- İşlenen sentetik exact-UTC maç: 5
- Power güncellemeleri sıfır toplamlıdır.
- Season carry kapalıdır: power_carry=0.
- H/D/A olasılıkları toplamı 1 ve expected-score kimliği korunur.
- Goal margin katmanı kapalı: bütün G değerleri 1.0.
- Achievement Reserve kapalı: bütün reserve ekleri 0.0.
- Tahmin çıktısı retrospective replay'dir; holdout kanıtı değildir.
- Final state takım sayısı: 10
