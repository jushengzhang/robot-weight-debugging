I’ll separate BOS, newline, and content-token attention rather than treating all prompt attention as useful. I’ll also check the size and direction of the action change; a large undirected action disturbance would argue against this edit.

```python
pr5=json.loads((W/"probe_5"/"results.json").read_text())
print("arms:",[(r["idx"],r["arm"]) for r in pr5])
def attention_metrics(r,l):
    a=np.load(workpath(r["attention"][str(l)])).mean(axis=(0,1))
    nv=r["prefix_layout"]["n_lang_valid"]
    return np.array([a[:256].sum(),a[256:512].sum(),a[768],
                     a[769:768+nv-1].sum(),a[768+nv-1],1-a.sum()])
by5=collections.defaultdict(dict)
for r in pr5: by5[r["idx"]][r["arm"]]=r
for idx,rs in by5.items():
    print("input",idx,rs.keys())
    arms=list(rs)
    for l in (6,9,12,15):
        print(" L",l, {a:[round(float(v)*100,2) for v in attention_metrics(rs[a],l)] for a in arms},
              "[base,wrist,BOS,content,newline,action]")
    if len(arms)==2:
        a=np.load(workpath(rs[arms[0]]["action"])); b=np.load(workpath(rs[arms[1]]["action"]))
        print(" action RMS change",round(float(np.sqrt(np.mean((a-b)**2))),4),
              "mean xyz shift",[round(float(v),4) for v in (b-a).mean(0)[:3]])
```
