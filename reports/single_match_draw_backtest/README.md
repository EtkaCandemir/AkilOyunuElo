# Single-Match Draw Backtest

Tek maçta tamamlanan eleme eşleşmeleri için format-duyarlı beraberlik
olasılığı çalışmasının paylaşılabilir sonuç paketidir.

- `backtest_report.md`: ana metodoloji, fold ve segment raporu.
- `selected_candidate.json`: istatistiksel ve mühendislik kararlarının ayrımı.
- `fold_selections.csv`: yalnız eğitim sezonlarından seçilen katsayılar.
- `fold_results.csv`: altı unseen sezon sonucu.
- `arm_summary.csv`: pooled baseline, walk-forward ve sabit `0.12` karşılaştırması.
- `segment_summary.csv`: turnuva, tek/çift maç ve COVID sonrası duyarlılık.
- `dependency_uncertainty.csv`: tie, takım-sezon ve ay bootstrap aralıkları.
- `candidate_surface.csv`: `0.04-0.24` parametre yüzeyi.

Yeniden üretim:

```bash
python3 scripts/run_single_match_draw_backtest.py --bootstrap-samples 4000
```
