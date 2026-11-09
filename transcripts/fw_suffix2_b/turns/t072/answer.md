Half dose lowers clean drift, but the full dose gives the stronger overall correction; neither eliminates the paraphrase outlier. I will test the full combined candidate on the 96 development episodes to see whether its improved action match translates into task success.
```python
update_hybrid_projection(rank=256,dose=1.)
import hashlib
practice=json.loads((Path(BENCH)/"practice_variants.json").read_text())
used={(v["suite"],v["id"]) for vs in sp.values() for v in vs}
lang_dev=[]
for suite in ["libero_10","libero_goal","libero_object","libero_spatial"]:
    pool=[v for v in practice if v["category"]=="Language Instructions" and v["suite"]==suite
          and (v["suite"],v["id"]) not in used]
    pool.sort(key=lambda v:hashlib.sha256(f'language-development-26:{v["suite"]}:{v["id"]}'.encode()).hexdigest())
    assert len(pool)>=24
    lang_dev.extend(pool[:24])
for i,v in enumerate(lang_dev):
    v=dict(v);v["seed"]=26000+i;lang_dev[i]=v
(W/"language_development.json").write_text(json.dumps(lang_dev,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\nPass25 hybrid half dose: executed gap ratio.7597, normalized.7754, normalized arm-only.8903; clean executed drift.00454, language.02830. Full dose24 normalized arm-only.8996 but improves gripper more (overall.6856). Next pass26 full hybrid development96 raw. New language_development.json holds96 unused paraphrase variants (24/suite), disjoint from every reserved split, for later development safety.\n")
print("CANDIDATE", (W/"current_candidate.json").read_text(), "LANGUAGE DEV",len(lang_dev))
```
