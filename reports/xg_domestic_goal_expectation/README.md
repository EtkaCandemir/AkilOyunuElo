# xG Form Terimi Domestic Attack/Defence Uzerinde

Karar: **`KEEP_SHADOW_CANDIDATE`.** Production contract'a dokunulmadi,
aktivasyon yapilmadi.

## Soru

Onceki kosu (`reports/xg_goal_expectation/`) xG form terimini **Elo-only** bir
tabana ekledi ve xG'nin gol kontrolunu gectigini gosterdi. O raporun kendi
yazdigi sinir suydu:

> Bu kosunun tabani reponun mevcut `DOMESTIC_ATTACK_DEFENCE_POISSON`
> kolundan daha basittir. Gosterilen sey "xG basit bir tabana eklendiginde
> yardim eder"dir; "xG mevcut en iyi modele eklendiginde yardim eder"
> **gosterilmemistir**.

Bu kosu tam olarak o bosluğu kapatir. Form terimi domestic attack/defence
modelinin **uzerine** konur, yerine degil:

| Kol | Ne | Rol |
|---|---|---|
| `DOMESTIC_AD` | reponun en iyi skor kolu | taban |
| `DOMESTIC_AD_GOALS_FORM` | ayni kol + gol form terimi | kontrol |
| `DOMESTIC_AD_XG_FORM` | ayni kol + xG form terimi | aday |

Gol kontrolu kaldirilmadi. Iki aday kol da **ayni iki ek parametreyi** tasir;
aralarindaki fark yalniz form teriminin okundugu kaynaktan gelebilir.

Ayni `3528` mac, ayni dort fold (`2022/23`-`2025/26`), ayni dort metrik.
Ornek, onceki kosununkiyle birebir aynidir; iki calisma dogrudan ust uste
konabilir.

## Sonuc

| Kol | exact_score_nll | Tabana fark |
|---|---:|---:|
| `DOMESTIC_AD` | 3.021498 | 0 |
| `DOMESTIC_AD_GOALS_FORM` | 3.021068 | -0.000430 |
| `DOMESTIC_AD_XG_FORM` | **3.019349** | **-0.002149** |

xG kolu hem tabani hem gol kontrolunu gecer. Exact-score fold kaydi: tabana
karsi `3/4`, gol kontrolune karsi `3/4`.

### Kazancin ne kadari hayatta kaldi

Ayni `3528` mac uzerinde iki calisma yan yana:

| Model | exact_score_nll |
|---|---:|
| `ELO_ONLY` (onceki kosu) | 3.029843 |
| `ELO_ONLY` + gol formu | 3.029642 |
| `ELO_ONLY` + xG formu | 3.027008 |
| `DOMESTIC_AD` (bu kosu) | 3.021498 |
| `DOMESTIC_AD` + gol formu | 3.021068 |
| `DOMESTIC_AD` + xG formu | **3.019349** |

Bu tablonun tasidigi asil bilgi sudur: xG formunun kazanci basit tabanda
`-0.002834`, domestic taban uzerinde `-0.002149`. Yani kazancin yaklasik
**`%76`'si hayatta kalir**. `XG_PRESENT` segmentinde ayni oran `%73`
(`-0.006970` -> `-0.005104`).

Domestic attack/defence katmani ile xG formu buyuk olcude **ayni bilgiyi
tasimiyor**. Onceki kosudaki kazancin domestic modelin zaten bildigi seyin
tekrari oldugu hipotezi elenmistir.

### Conservative envelope, tabana karsi

`exact_score_nll`, dependency-robust bootstrap, uc kumeleme goruşunun en genis
alt/ust siniri:

| Kol | Segment | Fark | %95 CI | Guvenilir |
|---|---|---:|---:|:--:|
| `XG_FORM` | `PHASE:MAIN` | -0.006106 | [-0.009441, -0.002732] | **evet** |
| `XG_FORM` | `XG_PRESENT` | -0.005104 | [-0.008347, -0.001876] | **evet** |
| `XG_FORM` | `ALL` | -0.002149 | [-0.005292, +0.000502] | hayir |
| `XG_FORM` | `PHASE:QUALIFYING` | +0.002345 | [-0.000732, +0.005494] | hayir |
| `XG_FORM` | `XG_ABSENT` | +0.002066 | [-0.001321, +0.005801] | hayir |
| `GOALS_FORM` | herhangi | — | — | **hicbiri** |

Gol kontrolunde hicbir segment guvenilir degildir. Guvenilir zarar hicbir
kolda, hicbir segmentte yoktur.

### Conservative envelope, xG kolu gol kontrolune karsi

Bu, kontrolun var olma sebebidir. Iki kol ayni parametre sayisini tasidigi
icin buradaki guvenilir bir fark yalniz **kaynaga** atfedilebilir:

| Segment | Fark | %95 CI | Guvenilir |
|---|---:|---:|:--:|
| `PHASE:MAIN` | -0.004385 | [-0.007279, -0.001362] | **evet** |
| `XG_PRESENT` | -0.003796 | [-0.006653, -0.001064] | **evet** |
| `ALL` | -0.001719 | [-0.004352, +0.000659] | hayir |
| `PHASE:QUALIFYING` | +0.001309 | [-0.002695, +0.004509] | hayir |
| `XG_ABSENT` | +0.001244 | [-0.003015, +0.004747] | hayir |

Kazanc iki ek parametreden degil, xG'den gelir.

## Taban gercekten reponun en iyi kolu mu

Aday zayiflatilmis bir tabani gecmis olsa sonuc anlamsiz olurdu. Bu yuzden
taban, yayinlanmis kolla ayni maclarda karsilastirilir:

| Kaynak | Mac | exact_score_nll |
|---|---:|---:|
| Yayinlanmis `DOMESTIC_ATTACK_DEFENCE_POISSON`, tam ornek | 4884 | 3.016594 |
| Yayinlanmis kol, bu kosunun ornegi | 3528 | 3.021437 |
| Bu kosunun `DOMESTIC_AD` tabani | 3528 | **3.021498** |

Fark `+0.000061` nats. Iki model bit bazinda ayni degildir - yayinlanmis kol
regularizasyonunu `1X2` hedefinde secer, bu kosu gol NLL'sinde secer - fakat
ayni modeldir ve ayni gucte kosar. `tests/test_xg_domestic_goal_model.py`
ayrica `none` kolunun `fit_european_poisson_transfer` ile ayni katsayilari ve
ayni oranlari urettigini pinler.

## Egitim penceresi duyarliligi

`sensitivity_xg_training_window/` ayni testi daha dar bir egitim penceresiyle
tekrarlar: yalniz xG kapsanan sezonlar (`2020/21`+). Boylece gol ve xG kollari
birebir ayni satirlari gorur.

| | Ana kosu (`full`) | Duyarlilik (`xg`) |
|---|---:|---:|
| `DOMESTIC_AD` | 3.021498 | 3.023138 |
| `+ gol formu` | -0.000430 | **+0.000354** |
| `+ xG formu` | -0.002149 | -0.002321 |
| xG vs gol, `XG_PRESENT` CI | [-0.006653, -0.001064] | [-0.008723, -0.001375] |
| xG vs gol, exact-score fold | 3/4 | **4/4** |

Sonuc iki pencerede de aynidir; dar pencerede xG lehine biraz daha gucludur ve
gol kontrolu orada tabana **zarar verir**. Ana kosu olarak genis pencere
secilmistir cunku tabani reponun bildirdigi gucte tutar.

## Uc onemli sinir

**1. Pooled `ALL` hala guvenilir degil.** Kazanc xG'nin bulundugu yerde
yogunlasir; ornegin `%41`'inde xG yoktur ve orada kol hafifce zarar verir
(`+0.002066`). Katman kosullu olmalidir.

**2. Kosullandirma kurali test edilmedi.** `XG_ABSENT` zarari, mevcut macin
xG'si olmamasindan gelmez - form terimi zaten yalniz gecmisi okur. Bu maclarin
buyuk cogunlugu on eleme turlaridir ve oradaki takimlarin **xG gecmisi**
incedir. Dogru kapi "mac xG tasiyor mu" degil, "takimin xG gecmisi yeterli mi"
olmalidir. Bu kosu o kapiyi denemedi.

**3. Turev pazarlar hala gecmiyor.** Ust 2.5 Brier `-0.000213`, CI
`[-0.001023, +0.000491]`; KG var `-0.000386`. Hicbir segmentte guvenilir
iyilesme yok. Skor katmanini `Diagnostic`'e koyan asil gerekce -
climatology'yi zar zor gecen ust/alt ve KG var - bu kosuyla **degismemistir**.

## Bu, skor katmanini `Diagnostic`'ten cikarir mi

Kismen. RESEARCH_STATUS `§7`'nin 6. maddesi olarak sorulan soru - "xG form
terimi reponun en iyi skor kolunun uzerinde de ise yarar mi" - **evet** olarak
yanitlanmistir, hem tabana hem kontrole karsi, iki egitim penceresinde de.

Fakat katmanin tamamini `Diagnostic`'ten cikarmaya yetmez, cunku o statuyu
doguran iki gerekceden yalniz biri kalkmistir:

- ~~"Skor katmani takim formunu hic tasimiyor"~~ -> kalkti.
- ~~"xG kazanci yalniz basit bir tabanda gosterildi"~~ -> kalkti.
- "Ust/alt ve KG var climatology sinirinda" -> **duruyor**.
- "Pooled envelope sifiri kesiyor" -> **duruyor**.

`PROMOTE_CANDIDATE` icin eksik olan iki sey nettir: gecerli bir kosullandirma
kurali ve turev pazarlarda olculebilir bir kazanc.

## Siradaki adim

xG gecmisi esigine gore kosullandirilmis bir kol: form terimi yalniz her iki
takimin da penceresinde yeterli xG gecmisi varken devreye girsin, aksi halde
sifirlansin. `XG_ABSENT` zarari yapisal olarak kapanirsa pooled envelope da
kapanabilir.

## Yeniden uretim

```bash
python3 scripts/run_xg_domestic_goal_expectation_backtest.py
```

Duyarlilik kolu:

```bash
python3 scripts/run_xg_domestic_goal_expectation_backtest.py --training-window xg --output-root output/xg_domestic_goal_expectation_backtest_2020_2026/sensitivity_xg_training_window
```

Onkosul: `data/xg_2020_2026/`, `data/domestic_league_matches_2013_2026/`,
`output/ml_1x2_backtest_2018_2026/pre_match_feature_store.csv` ve
`output/domestic_poisson_backtest_2018_2026/` ciktilari.
