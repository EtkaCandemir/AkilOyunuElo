# Aktif Model Algoritmalari

Bu belge aktif production davranisini adim adim tanimlar. Denklemlerin kompakt
referansi icin [FORMULAS.md](FORMULAS.md) kullanilmalidir.

## 1. Hedef Sezon ve Bes Sezon Penceresi

`season`, AO First Elo'nun uretildigi hedef sezondur. Static input'taki `t`,
hedef sezondan once tamamlanan en yakin sezondur; `t_minus_4` en eski sezondur.
Agirliklar eskiden yeniye:

```text
0.07, 0.13, 0.20, 0.27, 0.33
```

Domestic Surprise history kolonlari ayni bes tamamlanmis sezonu
`history_position_t_minus_5 ... history_position_t_minus_1` adlariyla tasir.
Isimlendirme farkli olsa da zamansal anlam aynidir: hedef sezon sonucu history
ortalamasina giremez.

## 2. AO First Elo Algoritmasi

### 2.1 Input validation

1. Dort input tablosunun zorunlu kolonlarini kontrol et.
2. `teams.team_id`, `country(season,country_code)`,
   `domestic(season,team_id)` ve `club(season,team_id,country_code)`
   duplicate anahtarlarini reddet.
3. Tum takimlar icin exact `season + team_id + country_code` club-history satiri
   ara. Avrupa gecmisi yoksa satir yine bulunmali ve bes sezon acik sifir
   olmali.
4. Sayisal degerlerin finite ve semantik aralikta oldugunu kontrol et.
5. Boolean alanlarda yalniz tanimli true/false ve 0/1 sozlesmesini kabul et.
6. Lig pozisyonu varsa takim sayisinin en az iki, pozisyonun tam sayi ve
   `1..N` icinde oldugunu kontrol et.
7. `is_league_champion=true` ve bilinen pozisyon `!=1` ise hata ver.

### 2.2 Country strength

1. Bes ulke puanini sezon agirliklariyla topla.
2. Benchmark `25` ile log normalize et.
3. Aktif `country_tail_beta=0` nedeniyle normalize degeri `1` uzerinde cap et.
4. `gamma=0.80` kuvvetiyle League Strength uret.

Bu sinyal takimin kendi Avrupa performansi degil, geldigi ligin/ulkenin Avrupa
gucu sinyalidir.

### 2.3 Domestic achievement

1. Lig sirasi biliniyorsa position percentile hesapla.
2. Percentile'i floor/scale egrisine cevir.
3. Sampiyonluk bayragi varsa finish score'u en az `1.00` yap.
4. Lig sirasi bilinmiyorsa normal finish score `0.10`; sampiyonsa `1.00` kullan.
5. Kupa kazananina cup base `0.62` ver.
6. Yalniz lig ve kupa birlikte kazanildiysa `0.08 * league_finish_score`
   double bonusu ekle.
7. `max(league_finish, cup_base) + double_bonus` degerini safety cap `1.10`
   altinda tut.

### 2.4 Domestic Prior

League Strength, domestic basarinin rating etkisini de olceklendirir. Ayni
lig sirasi guclu ligde daha fazla bilgi tasir. Base + league component +
achievement component toplanir.

### 2.5 European Prior

1. Kulubun kendi bes sezonluk Avrupa puanlarini agirlikli topla.
2. Benchmark `20` ile log normalize et.
3. Aktif `european_tail_beta=0` nedeniyle normu `1` uzerinde cap et.
4. Base `500` uzerine European boost ekle.

Country score ile club score ayni sey degildir. Birincisi lig gucu, ikincisi
takimin kendi Avrupa kanitidir.

### 2.6 European Exposure

Iki kanit miktari hesaplanir:

- Season exposure: bes sezonun kacinda oynandigi.
- Match exposure: her sezondaki `min(1, matches/match_cap)` degeri.

Bu ikisi `0.60/0.40` ile birlestirilir. Ham exposure aciklama/rating source icin
korunur. Rating blend'inde aktif tavan nedeniyle effective exposure en fazla
`0.85` olur. Boylece tam Avrupa gecmisinde bile Domestic Prior'in en az `%15`
agirligi kalir.

### 2.7 Prior blend

Domestic Prior ile European Prior arasinda effective exposure kadar ilerle.
Exposure sifirsa sonuc Domestic Prior; `0.85` ise `%15 domestic + %85 European`
olur.

### 2.8 Domestic Surprise

1. Guncel lig pozisyonunu dogrudan `0..1` percentile'a cevir.
2. Onceki bes sezonun percentile ortalamasini `0.07..0.33` ile hesapla.
3. Agirlikli varyans ve volatiliteyi hesapla.
4. Oynak gecmiste surprise etkisini `consistency` ile azalt.
5. Guncel percentile eksi tarihsel ortalamayi coefficient `0.40` ile Domestic
   Prior Elo birimine cevir.
6. Domestic adjustment'i `+/-30` ile sinirla.
7. AO First'e yansiyan kismi `(1-effective exposure)` ile azalt.

Bes tam history sezondan biri eksikse katman `INSUFFICIENT_HISTORY` olur ve
duzeltme sifirdir. Varyans isareti degistirmez; yalniz genligi azaltir.

### 2.9 Output ve ranking

Output invariantlari calistirilir. Sonra `ao_first_elo` azalan, `team_id` artan
sira ile deterministik olarak siralanir. Ratingler 500 veya 2000'e kirpilmaz.

## 3. Sezon Baslatma Algoritmasi

1. Her seed'in `team_id` ve AO First Elo degerini validate et.
2. Duplicate team seed'i reddet.
3. Aktif carry `0` oldugu icin `power_elo=ao_first_elo` ata.
4. Achievement Reserve ve uc progression bakiyesini sifirla.
5. Processed match/tie setlerini ve chronology pointer'larini bos baslat.

Ucuncu adim sezonlar arasi carry'dir. Sezon icindeki qualifier akisi ayridir
ve asagidaki adimlar sezon baslatmanin degil, her macin parcasidir:

1. Round'u `Q1/Q2/Q3/QUALIFYING_PLAYOFF/MAIN` anahtarina esle.
2. Base tur onemini `0.40/0.55/0.70/0.85` olarak sec.
3. Qualifier macinda `%50` retention'i ayni update'e gomerek efektif
   `K` carpanini `0.20/0.275/0.35/0.425` yap; MAIN icin `1.00` kullan.
4. UCL/UEL/UECL arasinda dusen takimda ayni `club_id` Power state'ini koru.
5. Ilk `MAIN` macindan once Power state'ini degistirme. Ayrica carry veya reset
   uygulama; dogrudan katilan takim da ayni mac-disinda-degismez kuralina tabidir.

## 4. Pre-Match Tahmin Algoritmasi

Her fixture icin sonuc gorulmeden once:

1. State model/config fingerprint uyumunu kontrol et.
2. Match ID daha once islenmis mi kontrol et.
3. Iki takim state'te var mi kontrol et.
4. `kickoff_utc + match_id` chronology gerilemesini reddet.
5. Home ve Away `AO Live Elo` degerlerini oku.
6. Neutral degilse global `H` ekleyerek `E_home` hesapla.
7. Fixture tek maclik knockout tie ise draw intercept `0.12`, degilse `0.24`
   sec.
8. Beklenen puani koruyan Home/Draw/Away olasiliklarini hesapla.
9. Ratingler, olasiliklar, config ID ve state pointer'lariyla prediction lock
   olustur.

`is_single_match_tie` sonuc, uzatma veya penalty metadata'sindan turetilmez.
Fikstur formati sonuc bilinmeden once verilir.

### 4.1 Production 1X2 Ensemble

Base AO prediction lock'tan sonra ve sonuc gorulmeden once:

1. Frozen Structural Logistic artifact'iyla ham ML olasiligi uret.
2. Ham ML'yi AO ile log-probability uzayinda `%90/%10` birlestir.
3. Iki kulubun causal Domestic Poisson snapshot'ini oku; `rho=0` ile ham
   Poisson 1X2 uret.
4. Ham Poisson'u AO ile `%50/%50` log blend yap.
5. Current ML ve AO Poisson bilesenlerini `%50/%50` log blend yap.
6. Tum ara ve final olasiliklari, coverage, artifact hash'leri ve fallback
   durumu ile pre-match loga yaz.

Artifact yuklenemezse veya satirin ML feature/state verisi gecersizse final
olasilik birebir AO 1X2 olur. Bu akis prediction-only'dir; sonucu settlement
kernel'ine veya AO Live Elo state'ine geri yazmaz.

## 5. Match Settlement Algoritmasi

### 5.1 Field score

Skor 90 dakika, uzatma oynandiysa 120 dakika saha skorudur. Shootout golleri
eklenmez.

```text
home_goals > away_goals -> S_home=1
home_goals = away_goals -> S_home=0.5
home_goals < away_goals -> S_home=0
```

`decided_on_penalties=true` saha sonucunun yonunu degistirmez. Ornegin ikinci
ayagi 2-0 kazanip aggregate'i esitleyen takim icin saha sonucu galibiyettir;
shootout yalniz ek GD/xG sinyallerini kapatir.

### 5.2 Controlled goal difference

- Draw, shootout karari veya tek farkta multiplier `1`.
- Iki ve daha fazla farkta `ln(GD)` sinyali kullan.
- GD'yi 4'te cap et.
- Pre-match effective rating farki buyudukce bonusu exponential olarak sondur.

Bu nedenle denk takimlar arasindaki 3-0, klasik sonucu daha cok buyutur; agir
favorinin 5-0'i cap ve damping nedeniyle sinirli kalir.

### 5.3 xG performance

xG yalniz su kosullarda uygulanir:

- iki takim xG'si birlikte mevcut,
- satir analysis-eligible,
- field result draw degil,
- match shootout ile karara baglanmamis.

Home perspective xG farki `tanh` ile `[-1,1]` sinyaline doner. Sinyal klasik
result residual buyuklugunun en fazla `%30`u kadar eklenir veya cikarilir.
Sonuc Delta'sinin yonu korunur. Iki xG birlikte yoksa imputation yapilmaz ve
goal-margin-only fallback kullanilir.

`minimum_winner_gain_ratio=0.70`, aktif `%30` bounded oraninin analitik
sonucudur. Runtime floor'un normal kosullarda ayrica devreye girmesi beklenmez.

### 5.4 Power update

Birlesik residual `K` ile carpilarak Delta uretilir:

```text
Home Power Post = Home Power Pre + Delta
Away Power Post = Away Power Pre - Delta
```

Her macta toplam fark `1e-9` toleransiyla sifir olmalidir.

### 5.5 Tie ve progression

Tie ilk goruldugunde iki takim, competition, stage ve pre-tie neutral expected
advance proxy'si kaydedilir. `is_tie_decider=true` oldugunda:

1. `advanced_team_id` tie taraflarindan biri mi kontrol et.
2. Ayni `tie_id` daha once tamamlanmis mi kontrol et.
3. Stage `ROUND_OF_16`, `QUARTERFINAL`, `SEMIFINAL` veya `FINAL` ise fixed bonus
   uygula.
4. UCL/UEL/UECL oranini `1 : 2/3 : 1/3` kullan.
5. Competition sezon cap'ini asma.
6. Kazananin ilgili progression bakiyesini artir; kaybedenden dusme.
7. Tie'yi processed olarak isaretle.

Knockout play-off, lig asamasi ve on elemeler bonus uretmez. Penalty ile tur
atlayan takim field score kurallarina gore Power update alir ve tie tamamlandigi
icin progression bonusunu tam alir.

## 6. Exact-UTC Batch Kurali

Ayni `kickoff_utc` degerindeki butun maclar tek snapshot'tan tahmin edilmelidir.
Guvenli batch akisi:

1. Batch basindaki state'i dondur.
2. Tum fixtures icin prediction lock uret.
3. Tum prediction kayitlarini yaz.
4. Sonuclari deterministik `match_id` sirasiyla settle et.

Bir macin sonucu ayni anda baslayan baska macin pre-match tahminine giremez.

## 7. Idempotency ve Replay

- Ayni `match_id` ikinci kez islenemez.
- Chronology geriye gidemez.
- Locked prediction state/config ile uyusmazsa settlement reddedilir.
- Ayni input, config ve siralama ayni sayisal sonucu uretmelidir.
- Replay scripti production kernel'ini kullanmali; formulu yerel olarak yeniden
  implement etmemelidir.
