# Served ML/Poisson Layer Under the Participation-Normalized Seed

Karar: **`PASS`.** Servis edilen katman aday seed altinda hem guvenli hem de
katki uretmeye devam ediyor. Production contract degistirilmedi.

## Soru

`european_participation` calismasi seed'i `1008 / 1887` kulup-sezonda oynatiyor.
Servis edilen `%50 Current ML + %50 AO Domestic Poisson` blend'i ise agirliklarini
production seed'i uzerinde secmisti. Structural Logistic feature'lari AO
log-odds'unu, AO First/Live rating farklarini ve exposure'i okudugu icin katman
yeni seed altinda otomatik olarak guvenli sayilamaz.

Calismanin kendi "Sinirlar" bolumu bunu acikca birakmisti: *"Servis edilen
ML/Poisson katmani yeniden secilmedi. Terfi oncesi candidate-seed-aware bir
served replay gerekir."* Bu paket o replay'dir.

## Yontem

Iki kol, her biri icin **tam zincir bastan kuruldu**:

```text
seed -> mac replay -> pre-match feature store -> ML backtest
     -> Domestic Poisson backtest -> ML/Poisson ensemble
```

- `control`: servis edilen production seed'i.
- `candidate`: katilim-normalize seed, `k = 0.20` (modal fold secimi).

Kontrol kolu servis edilen seed'i `4.5e-13` Elo icinde yeniden uretir ve
`CURRENT_AO` Brier'i `0.569287` cikar - canonical evaluation'in ilan ettigi
degerin aynisi. Zincir bu sekilde dogrulanmistir.

Feature store'un **statik** seed kolonlari da kol bazinda yeniden yazilir
(`home_initial_rating`, `away_initial_rating`, `initial_rating_difference`).
Ilk kurulumda bu kolonlar production seed'inde kalmisti; o halde iki kol AO
olasiligi uzerinde ayrisirken onu ureten seed uzerinde ayni gorunuyordu.
`effective_european_exposure` bilerek degistirilmez - katman exposure
agirligina dokunmaz.

## Sonuc

`4884` unseen mac:

| model | control Brier | candidate Brier | fark |
|---|---:|---:|---:|
| `CURRENT_AO` | 0.569287 | 0.566512 | **-0.002775** |
| `CURRENT_ML_BLEND` | 0.565544 | 0.562734 | -0.002810 |
| `AO_POISSON_BLEND` | 0.566742 | 0.564292 | -0.002449 |
| `AO_POISSON_RHO0_CONTROL` | 0.566647 | 0.564178 | -0.002469 |
| `ML_POISSON_ENSEMBLE` | 0.564837 | 0.562152 | -0.002685 |

Butun bilesenler aday seed altinda iyilesiyor; hicbiri gerilemiyor.

### Asil kapi: servis edilen katman kendi AO tabanina karsi

```text
control    AO 0.569287 -> ensemble 0.564837   fark -0.004450
candidate  AO 0.566512 -> ensemble 0.562152   fark -0.004360
```

Katmanin kattigi deger aday seed altinda **neredeyse aynen korunuyor**
(`-0.00009` fark). Accuracy katkisi ise buyuyor: `+0.001433` -> `+0.002662`.

### Conservative envelope, ensemble vs kendi AO tabani

```text
scope              control                              candidate
ALL                -0.004450 [-0.006597,-0.002196] R    -0.004360 [-0.006475,-0.002228] R
competition:UCL    -0.004012 [-0.008797,+0.001098]      -0.004199 [-0.008939,+0.000741]
competition:UEL    -0.001387 [-0.004960,+0.002363]      -0.000607 [-0.004214,+0.003217]
competition:UECL   -0.006851 [-0.009963,-0.003920] R    -0.007051 [-0.010098,-0.004060] R
coverage:BOTH      +0.000207 [-0.003905,+0.004215]      -0.001499 [-0.005826,+0.002728]
coverage:ONE       -0.009002 [-0.014445,-0.003508] R    -0.008491 [-0.013489,-0.003428] R
coverage:NONE      -0.002636 [-0.004601,-0.000872] R    -0.001580 [-0.003976,+0.000503]
```

`R` = guvenilir iyilesme. **Iki kolda da hicbir segmentte `reliable_harm` yok.**
Segment yapisi iki kol arasinda korunuyor; tek kayda deger fark
`coverage:NONE`'in adayda sifiri kesmesi, ki bu Poisson kapsami olmayan
maclarda AO'nun zaten iyilesmesinin dogal sonucudur.

## Yan bulgu

Yayinlanmis `reports/production_prediction/` paketi `2026-08-21` exposure cap
yayilimindan once dondurulmustu ve `CURRENT_AO` icin `0.572093` tasiyordu. Bu
kosunun kontrol kolu ayni buyuklugu guncel contract'la `0.569287` olarak uretir;
servis edilen ensemble da `0.568093` yerine `0.564837` cikar. Eski paket
stale'dir.

## Sinirlar

- ML hiperparametreleri her kolda kendi feature store'u uzerinde yeniden
  secilir. Bu kasitlidir - production periyodik olarak yeniden egitir - fakat
  "ayni model, farkli girdi" degil "ayni prosedur, farkli girdi" karsilastirmasi
  oldugu anlamina gelir.
- Kosu `2018/19-2025/26` gelistirme penceresindedir; prospective kanit degildir.
- `k = 0.20` modal fold secimidir, tek dogru deger degildir.

## Yeniden uretim

```bash
python3 scripts/build_participation_candidate_feature_store.py
```

Ardindan her kol icin ML -> Poisson -> ensemble zinciri
`output/participation_served_ensemble_2018_2026/` altinda kosulur.
