The half-strength edit gives three paired recoveries and no regressions, but 17 episodes have a null termination in both arms, so I will inspect those records before treating the aggregate as reliable. I will also quantify whether the affected probes moved toward the intended task representations.
```python
r7=read_roll(7); e6=entries(read_probe(6))
print("Null termination examples:")
nulls=[r for r in r7 if r.get("termination") is None]
print("Count",len(nulls))
for r in nulls[:2]: print(json.dumps(r,ensure_ascii=True)[:1800])
def paired_records(rr):
    pairs={}
    for r in rr:
        key=(r["suite"],r["id"],r["seed"])
        pairs.setdefault(key,{})[r["model"]]=r
    return pairs
pairs7=paired_records(r7)
print("Model labels:",collections.Counter(r["model"] for r in r7))
for key, pair in pairs7.items():
    vals=list(pair.values())
    if len(vals)==2 and vals[0]["success"]!=vals[1]["success"]:
        print("Changed",key,[(r["model"],r["success"]) for r in vals],vals[0]["instruction"])
print("Probe arms:",[(e["idx"],e["arm"],input_id(e)) for e in e6[:5]])
def probe_by_id(k):
    return {(input_id(e),e["arm"]):e for e in entries(read_probe(k))}
pb6=probe_by_id(6)
for i,target in [(1420,1785),(1262,2473),(1179,2553),(1385,1785),(1785,1785),(2473,2473),(2553,2553)]:
    a=next(e for (j,arm),e in pb6.items() if j==i and arm=="orig")
    b=next(e for (j,arm),e in pb6.items() if j==i and arm!="orig")
    ao=np.load(a["action"]); ae=np.load(b["action"]); at=actions4[target]
    print("\nID",i,"action change RMS",round(float(np.sqrt(np.mean((ae-ao)**2))),4),
          "target xyz RMS orig/edit",
          round(float(np.sqrt(np.mean((ao[:5,:3]-at[:5,:3])**2))),4),
          round(float(np.sqrt(np.mean((ae[:5,:3]-at[:5,:3])**2))),4))
    za=np.load(a["activations"]); zb=np.load(b["activations"])
    vals=[]
    for l in [0,3,6,9,12,17]:
        key=f"mean::expert.{l}.resid::action"
        X=np.stack([acts4[j][key] for j in control_ids]); mu=X.mean(0)
        t=acts4[target][key]-mu
        cos=lambda x:float(np.dot(x-mu,t)/(np.linalg.norm(x-mu)*np.linalg.norm(t)+1e-8))
        vals.append((l,round(cos(za[key]),3),round(cos(zb[key]),3)))
    print("Expert target cosine L,orig,edit:",vals)
```
