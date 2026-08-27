# UCL/UEL Yerel Lig Poisson Kapsama Aday Değerlendirmesi

## Statü

Bu rapordaki loss tablolari candidate veri setiyle uretilmis tarihsel
degerlendirmedir; parametre secimi icin yeniden kullanilmaz. Veri/kimlik
kapsami, parametreler degistirilmeden, `2026-08-21-ucl-uel-domestic-poisson-coverage`
revizyonuyla production checkpoint'ine alinmistir. Aktif checkpoint
`artifacts/production_prediction/domestic_poisson_state_2026_27.json` olup
yalniz Avrupa maci oncesindeki causal yerel sonuclari icerir.

## Kapsam

- Hedef UCL/UEL kulübü: **80**
- Candidate state uygun: **79**
- Açıklanmış fallback: **1**
- Candidate domestic fixture: **24586** hedef-kulüp görünümü
- Production contract SHA-256: `8bd87e88dab505c6177cbe20bd6a15fb6cc2a3a5fc1b005582519f995dd33e57`

## Pooled model karşılaştırması

```csv
model,matches,brier_1x2,log_loss_1x2,accuracy_1x2,delta_vs_ao_brier_1x2,delta_vs_ao_log_loss_1x2,delta_vs_ao_accuracy_1x2
CURRENT_AO,4884,0.5720926654555982,0.9643708372415034,0.5501638001638002,0.0,0.0,0.0
AO_POISSON_BLEND,4884,0.5718287525948811,0.9643551483155444,0.5532350532350533,-0.0002639128607171415,-1.5688925959045363e-05,0.0030712530712531105
AO_ML_POISSON_BLEND,4884,0.5674334852287891,0.958379161671955,0.5561015561015561,-0.004659180226809112,-0.0059916755695483825,0.0059377559377559175
```

## Fold sonuçları

```csv
fold,test_season,model,matches,brier_1x2,log_loss_1x2,accuracy_1x2,baseline_brier,baseline_log_loss,baseline_accuracy,delta_brier_vs_ao,delta_log_loss_vs_ao,delta_accuracy_vs_ao
1,2020/21,AO_ML_POISSON_BLEND,540,0.5296327831930847,0.9005782988482076,0.5944444444444444,0.5296327831930847,0.9005782988482077,0.5944444444444444,0.0,-1.1102230246251565e-16,0.0
2,2021/22,AO_ML_POISSON_BLEND,816,0.5762709483211054,0.9711081738467205,0.5453431372549019,0.5762709483211054,0.9711081738467205,0.5453431372549019,0.0,0.0,0.0
3,2022/23,AO_ML_POISSON_BLEND,804,0.580081626067044,0.9762802387522636,0.5323383084577115,0.5922543302028854,0.9933671776809765,0.5149253731343284,-0.012172704135841372,-0.017086938928712936,0.01741293532338306
4,2023/24,AO_ML_POISSON_BLEND,806,0.5574092365210123,0.9466785743741373,0.5744416873449132,0.5642390493784643,0.9534484792111946,0.5558312655086849,-0.006829812857451989,-0.006769904837057306,0.018610421836228297
5,2024/25,AO_ML_POISSON_BLEND,957,0.5677086115769593,0.9597437942300022,0.5559038662486938,0.5692229584595572,0.9607593419315188,0.5642633228840125,-0.0015143468825978834,-0.0010155477015165726,-0.008359456635318674
6,2025/26,AO_ML_POISSON_BLEND,961,0.5787218853591354,0.9735278047745942,0.5483870967741935,0.5849804940716918,0.9829940088682142,0.5400624349635796,-0.006258608712556457,-0.009466204093619979,0.008324661810613865
```

## Olasılık farkı

```csv
model,matches,mean_absolute_probability_shift
AO_ML_POISSON_BLEND,4884,0.010481370547873936
AO_POISSON_BLEND,4884,0.023711738626732478
CURRENT_AO,4884,0.0
```

## Not

Yeni veri, mevcut 19 ligdeki kaynak satırlarını değiştirmez. Primary/secondary seçimleri league-season seviyesinde tek kaynaktan yapılır; kaynak değişen hedef liglerde state anahtarları yalnız doğrulanmış AO kulüpleri için kanonikleştirilir.
