The reserved evaluation is complete. I will pool each variant once, replace the zero-step failures with their completed retries, and audit whether prompts containing no edited tokens remain exactly unchanged. Then I will test all six non-language categories.
```python
confirmed={}
for k in [20,21,22,23]:
    confirmed.update(pair_roll(k))
assert len(confirmed)==460
assert all(r["termination"] in ("success","max_steps") for p in confirmed.values() for r in p.values())
print("INDEPENDENT F CONFIRMATION",paired_stats(confirmed))
for field in ["suite","difficulty"]:
    for g in sorted({p["original"][field] for p in confirmed.values()}):
        z={key:p for key,p in confirmed.items() if p["original"][field]==g}
        print(field,g,paired_stats(z))
touched={}
untouched={}
for key,p in confirmed.items():
    (touched if set(sp.encode(p["original"]["instruction"])) & set(F["rows"]) else untouched)[key]=p
print("Touched",paired_stats(touched),"Untouched",paired_stats(untouched))
print("Untouched differing success/steps",sum(
    (p["original"]["success"],p["original"]["steps"]) != (p["edited"]["success"],p["edited"]["steps"])
    for p in untouched.values()))
trace_checks=[]
for key,p in list(untouched.items())[:16]:
    paths=[pathlib.Path(p[m]["actions_npy"]) for m in ["original","edited"]]
    if all(q.exists() for q in paths):
        a,b=[np.load(q) for q in paths]
        trace_checks.append((a.shape==b.shape,bool(np.array_equal(a,b))))
print("Untouched trace audit",collections.Counter(trace_checks))
statsF=paired_stats(confirmed)
json.dump({"overall":statsF,"episodes":[{"suite":s,"id":i,"seed":seed,
           "original":p["original"]["success"],"edited":p["edited"]["success"]}
           for (s,i,seed),p in confirmed.items()]},(W/"F_confirmation_results.json").open("w"),indent=2)
with (W/"notes.md").open("a") as f:
    f.write("\n## F independently confirmed (passes20-23)\nFrozen before these460 variants. "
            "All startup-error pairs rerun in pass23, each variant counted once. "+str(statsF)+
            "\nKeep F unchanged as final candidate. Next: paired safety checks for all six other categories; "
            "replicate independent language set on new seeds. best_F_dev/ is exact saved candidate.\n")
print("Nonlanguage categories",sorted({v["category"] for v in V if v["category"]!="Language Instructions"}))
```
