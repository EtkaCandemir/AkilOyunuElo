# Domestic Prior: lig gucu sekli

Karar: **`PROMOTE_CANDIDATE`** — aktive **EDILMEDI**. Production `2026-09-02`
itibariyla `gamma = 0.80`, `country_strength_benchmark = 25.0` ile calisiyor.

Bu paket bir aday kaydidir. Kilit (`2026-09-08`) oncesinde aktivasyon
onerilmemistir; gerekce en altta.

## Soru

Domestic Prior'in lig gucu bileseni:

```text
L = ( ln(1 + agirlikli_ulke_skoru) / ln(1 + benchmark) ) ^ gamma      1.0'da kirpilir
DP = 500 + 519.905 x L + 594.177 x A x (0.4 + 0.6 x L)
```

Iki sabit var ve ikisi de kanitla secilmemis:

- `benchmark = 25` v1.1'den devralindi. Sekiz sezon boyunca hicbir ulke ona
  degmedi (`2018/19`-`2025/26` maksimumlari `19.77`-`23.45`), yani kirpma olu
  koddu. `2026/27`'de Ingiltere `25.35` ile **tarihte ilk kez** asiyor.
- `gamma = 0.80` de v1.1'den devralindi. `run_ranking_first_calibration.py`
  onu taramisti ve alti fold'un altisi da `gamma >= 1.4` istemisti, ama
  fold'lar benchmark ve component'te dagildigi icin kapi hicbir seyi terfi
  ettirmedi ve `0.80` yerinde kaldi.

`gamma < 1`, zaten log ile sikistirilmis bir buyuklugu **ikinci kez**
sikistiriyor: Gurcistan ham `0.241` iken `L` olarak `0.320` aliyor, yani
Ingiltere ile arasindaki mesafenin dortte biri kapaniyor.

## Uc kosu

### 1. Tek eksen (`axis_*`)

`gamma`, `domestic_league_component` ve `domestic_achievement_component` ayri
ayri, digerleri uretimde sabit.

```text
gamma   Brier      dBrier     Spearman    dSpear
 0.8   0.565489   +0.000000   0.467299   +0.000000    URETIM
 1.2   0.563879   -0.001610   0.474885   +0.007586
 1.4   0.563750   -0.001739   0.475714   +0.008415    ic optimum
 1.6   0.563869   -0.001619   0.475369   +0.008070
 2.0   0.564538   -0.000951   0.473715   +0.006416
```

Iki bilesen her zamanki takasi verdi — `lig x1.30` Brier'i iyilestirip
siralamayi bozdu (`-0.001512` / `-0.002783`), `basari x0.85` tersini yapti.
**Yalniz `gamma` ikisini birden iyilestirdi.**

`gamma 1.4` kapiya takildi: UECL Brier `+0.000183`. Segment kapisi sifir
toleranslidir. O sayinin kendi guven araligini hesapladik:

```text
UCL   Brier -0.002030   CI [-0.003454, -0.000699]   sifiri disliyor
UEL   Brier -0.003148   CI [-0.005126, -0.001358]   sifiri disliyor
UECL  Brier +0.000183   CI [-0.001827, +0.002211]   sifiri ICERIYOR
```

Yani kapi gercek bir zarara degil, `2073` macta olculemeyen bir farka takildi.

### 2. Sabit benchmark x gamma (`fixed_benchmark_*`)

`benchmark` ve `gamma` ayni buyuklugu sekillendirdigi icin birlikte tarandi.
`PROMOTE_CANDIDATE` cikti, secilen `b15_g2.0`, alti fold da ayni.

**Bu aday reddedildi**, cunku nasil kazandigi kabul edilemez:

```text
benchmark 25 -> 1 ulke kirpilir (Ingiltere)
benchmark 15 -> 6 ulke kirpilir

b15'te kirpilanlar: England 1.1800, Spain 1.1049, Italy 1.1046,
                    Germany 1.0875, France 1.0437, Portugal 1.0193
                    -> hepsi 1.0000, aralarinda SIFIR ayrim
```

Optimizer, alti buyuk ligi tavana yigip geri kalani `gamma 2.0` ile ezerek
kazaniyor. Bu, `european_tail_beta`'nin `2026-08-30`'da kaldirilan kusurunun
ulke tarafindaki ikizi. Ayrica aday grid'in kosesinde ve iki eksende de egri
hala dusuyordu.

### 3. Sezona goreli cipa (`anchor_*`)

Sabit `25` yerine o sezonun referans ligi. Uc cipa denendi.

```text
aday          Brier      dBrier     Spearman    dSpear     kirpilan (8 sezon)
URETIM      0.565489   +0.000000   0.467299   +0.000000        -
max_g1      0.563853   -0.001635   0.472572   +0.005273        0
max_g1.2    0.563157   -0.002332   0.475160   +0.007861        0
max_g1.4    0.562851   -0.002638   0.476405   +0.009106        0
max_g1.6    0.562805   -0.002684   0.476932   +0.009633        0     <- onerilen
max_g2      0.563181   -0.002307   0.475017   +0.007718        0
top3_g1.6   0.562316   -0.003173   0.477426   +0.010127       13
p95_g1.6    0.561881   -0.003608   0.476445   +0.009146       24
p95_g2      0.561847   -0.003642   0.475967   +0.008668       24
```

Onbes varyantin **onbesi de** iyilestiriyor, yani bulgu cipa secimine kirilgan
degil. Ama Brier kirpma arttikca iyilesiyor: `p95` her sezon uc ulkeyi
(`2025/26`'da Ingiltere, Italya, Ispanya) tavana yiginca en dusuk Brier'i
veriyor. Ayni pathololoji, daha ilimli hali.

**`max` tek temiz cipa**: cipa maksimum oldugu icin hicbir sey onu asamaz,
kirpma yapisal olarak imkansiz, `country_tail_beta` anlamsizlasir.

## Onerilen aday: `max_g1.6`

```text
benchmark = o sezonun en guclu liginin agirlikli skoru
gamma     = 1.6

tam-tarih yuzeyinde, uretime gore:
  Brier      -0.002684
  Spearman   +0.009633
  ic optimum 1.6'da (2.0 daha kotu)
  kirpilan ulke: 0
```

Walk-forward tarafi (`relative_*`) su sayilari verdi. Bunlar **nested kolun**
sayilaridir, yani `max_g1.6`'nin tek basina degil: alti fold'un besi `g1.6`
sectiginde biri `g2.0` secti, ve bu satirlar o karisimin sonucudur.

```text
Brier      -0.002358   CI [-0.003652, -0.000993]   6/6 fold, zarf sifiri disliyor
log-loss   -0.003372   CI [-0.005177, -0.001554]   6/6 fold, zarf sifiri disliyor
UCL -0.003057   UEL -0.004906   UECL -0.000137     ucu de iyilesiyor
same_candidate_every_fold: False  (5 x g1.6, 1 x g2.0)
```

UECL'in isaret degistirmesi onemli: `gamma 1.4`'u sabit benchmark'ta engelleyen
kapi burada aciliyor.

Kiyas: `2026-08-30`'da terfi eden `european_tail_beta` degisikligi
`-0.001010` Brier ve `+0.000246` Spearman'di. Bu, Spearman'da **39 kati**.

### Cipa tek basina kazanc uretmiyor

`max_g0.8` (cipa degisir, gamma uretimde kalir): `-0.000341` Brier,
`-0.000607` Spearman. Neredeyse sifir.

Sebebi cebirsel: bir sezon icinde max-normalizasyon herkesi **ayni carpanla**
olcekler (`2025/26`'da `1.0154`), siralamayi hic degistirmez. Kazanc
`gamma`'dan gelir.

Cipanin katkisi baska: **`gamma`'yi yukseltmeyi guvenli kilar.** Sabit
benchmark'la yuksek gamma denendiginde optimizer dusuk benchmark'a kacip alti
ligi tavana yigiyordu. Max cipada o kacis yolu kapali.

## Bilinmesi gerekenler

- **Tek yonlu asagi cekme.** `gamma` yukselince `L = norm^gamma` asla artmaz.
  Cipa ligi disinda herkes duser: `2026/27`'de 108 kulubun 99'u, ortalama
  `-39.2` Elo. Tahmine zararsizdir (olasilik yalniz rating **farkina** bakar,
  ve yayilim `262 -> 283` artar), ama AO First yayinlanan bir sayiysa gorunur.
- **Cipa her sezon oynar.** Bir ligin puani yukselirse, kendi puani hic
  degismeyen bir lig normalize olarak duser. Sezonlar arasi mutlak
  karsilastirilabilirlik kaybolur. `power_carry = 0` oldugu icin ratingler
  sezonlar arasi tasinmiyor, yani bu tahmine yansimiyor.
- **Bu bir kod degisikligi, parametre degisikligi degil.**
  `country_strength_benchmark` su an config'de tek bir sayi. Sezon maksimumuna
  baglamak, pipeline'in ulke skorlarini okuyup cipayi hesaplamasini gerektirir.
- **Aktive edilirse** bu, sezon oncesi **dorduncu** AO First degisikligi olur
  (`participation` 08-27, `tail` 08-30, ve bu). Tam propagasyon zinciri +
  ledger revision 3 + belgeler.

## Neden kilit oncesi onerilmedi

Kilit `2026-09-08`, bu paket `2026-09-02`'de yazildi — alti gun. Aday saglam
ama aktivasyon zinciri (backtest zaten var; propagasyon, ledger, contract,
belgeler, PDF'ler) yarim birakilirsa canliya tutarsiz bir durum cikar. Ilk
plandaki kendi uyarimiz gecerli: *"yarim propagasyonla canliya cikmak en kotu
sonuctur."*

Aday kilit sonrasi, aceleye gelmeden aktive edilmelidir.

## Paketteki dosyalar

```text
axis_*                  tek eksen kosusu (gamma, iki bilesen)
fixed_benchmark_*       sabit benchmark x gamma (reddedilen b15_g2)
relative_*              max cipasi x gamma, walk-forward
anchor_*                max / top3 / p95 karsilastirmasi
anchor_low_gamma_*      gamma 1.0 ve 1.2 dolgusu
current_2026_27_max_g16.csv   108 kulubun max_g1.6 ile hali
```

`anchor_competition_summary.csv` ve `anchor_decision.json` o kosunun **secilen**
adayina (`p95_g2`) aittir, `max_g1.6`'ya degil. `max_g1.6`'nin walk-forward
sayilari `relative_*` dosyalarindadir.

## Uretim

```bash
python3 scripts/run_domestic_prior_axis_backtest.py --bootstrap-samples 4000
python3 scripts/run_country_shape_backtest.py --bootstrap-samples 4000
python3 scripts/run_relative_country_benchmark_backtest.py --bootstrap-samples 4000
python3 scripts/run_country_anchor_backtest.py --bootstrap-samples 4000
python3 scripts/run_country_anchor_low_gamma_backtest.py --bootstrap-samples 4000
```

Her kosu production contract'ini basta ve sonda hash'ler; besi de contract'in
degismedigini dogruladi.
