The value-only edit also had zero net tagged gain, though it gained one paraphrase episode. I’ll make one final dose test at half strength. This keeps the same directions and layers, testing whether the full correction is causing avoidable losses before I freeze a candidate for independent validation.
```python
helper = '''
from pathlib import Path
import json, numpy as np
W=Path(WORK)
def paired_stats(passes, category=None, exclude_language=False):
    pairs={}
    for k in passes:
        for line in (W/f"roll_{k}.jsonl").read_text().splitlines():
            r=json.loads(line)
            if category is not None and r.get("category")!=category: continue
            if exclude_language and r.get("category")=="Language Instructions": continue
            key=(k,r["suite"],r["id"],r["seed"])
            pairs.setdefault(key,{})[r["model"]]=r
    vals=[]
    for key,p in pairs.items():
        assert "original" in p and "edited" in p, (key,p.keys())
        vals.append((int(p["original"]["success"]),int(p["edited"]["success"])))
    a=np.array(vals,dtype=float)
    d=a[:,1]-a[:,0]; n=len(d)
    se=float(d.std(ddof=1)/np.sqrt(n)*100) if n>1 else float("nan")
    diff=float(d.mean()*100)
    return {"passes":passes,"n":n,"original":int(a[:,0].sum()),"edited":int(a[:,1].sum()),
            "wrong_to_right":int((d>0).sum()),"right_to_wrong":int((d<0).sum()),
            "diff_pp":diff,"se_pp":se,"ci95_pp":[diff-1.96*se,diff+1.96*se]}
'''
(W/"rollout_helpers.py").write_text(helper)
exec(helper)
for k in [25,37,38]:
    print("TAGGED",json.dumps(paired_stats([k],exclude_language=True)))
    print("LANGUAGE",json.dumps(paired_stats([k],category="Language Instructions")))
clear_edits()
for layer in range(30):
    n=f"mot.mixtures.video.blocks.{layer}.cross_attn.v.weight"
    o=load_original(n)
    save_edit(n,o-.5*(o@By)@Dy.T)
print("value-only dose .5 active tensors",len(edited()))
with (W/"notes.md").open("a") as f:
    f.write("\nPass38 value-only full development: tagged49->49/96 (0pp), language26->27/32 (+3.125pp). Pass39 same directions/layers at half dose, final development test. No validation variants used for choice. Remaining11 passes reserved for frozen-candidate independent raw/strip/language validation.\n")
```
