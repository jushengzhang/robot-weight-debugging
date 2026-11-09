Full replacement yields six recoveries and no regressions, versus three at half strength, although null episodes again reduce the usable sample. I will inspect the runtime records, summarize the dose-response, and prepare a broader but still token-sparse set of aliases for descriptive object phrases.
```python
def paired_summary(k, verbose=False):
    pp=paired_records(read_roll(k))
    valid=[p for p in pp.values() if set(p)=={"original","edited"} and
           all(r.get("termination") is not None for r in p.values())]
    ds=np.array([int(p["edited"]["success"])-int(p["original"]["success"]) for p in valid],float)
    out={"valid":len(valid),"total":len(pp),
         "orig":sum(p["original"]["success"] for p in valid),
         "edit":sum(p["edited"]["success"] for p in valid),
         "gain_pp":round(float(100*ds.mean()),2) if len(ds) else None,
         "se_pp":round(float(100*ds.std(ddof=1)/np.sqrt(len(ds))),2) if len(ds)>1 else None,
         "WR":int((ds>0).sum()),"RW":int((ds<0).sum())}
    if verbose:
        for p in valid:
            if p["original"]["success"]!=p["edited"]["success"]:
                r=p["original"]
                print("FIX" if p["edited"]["success"] else "BREAK",r["suite"],r["id"],r["instruction"])
    return out
print("Half:",paired_summary(7))
print("Full:",paired_summary(9,True))
pb8=probe_by_id(8)
for i,target in [(1420,1785),(1262,2473),(1179,2553)]:
    out=[]
    for dose,pb,arm in [(0,pb8,"orig"),(.5,pb6,"edit"),(1,pb8,"edit")]:
        entry=next(e for (j,a),e in pb.items() if j==i and (a=="orig")== (arm=="orig"))
        act=np.load(entry["action"]); za=np.load(entry["activations"])
        key="mean::expert.17.resid::action"
        mu=np.stack([acts4[j][key] for j in control_ids]).mean(0)
        x=za[key]-mu; t=acts4[target][key]-mu
        out.append((dose,round(float(np.dot(x,t)/(np.linalg.norm(x)*np.linalg.norm(t))),3),
                    round(float(np.sqrt(np.mean((act[:5,:3]-actions4[target][:5,:3])**2))),4)))
    print("Dose,target cosine,xyz RMS",i,out)
print("Runtime log candidates:",
      [str(p.relative_to(W)) for p in (W/"fwd/roll_9").glob("*")][:20])
# Potential descriptive aliases; reject ambiguous generic nouns for now.
candidate_aliases={**base_aliases,
 "cook":"stove", "cooking":"stove", 
 "rounded":"bowl", "round":"bowl", "mixing":"bowl",
 "flat":"plate", "forefront":"front",
 "central":"middle", "uppermost":"top", "upper":"top",
 "higher":"top", "highest":"top", "elevated":"top", "atop":"on",
 "framework":"rack", "holder":"rack",
 "spreadable":"cheese", "cultured":"cheese", "dairy":"cheese",
 "churned":"butter", "woven":"basket", "citrus":"orange",
 "broth":"soup", "condiment":"sauce", "ceramic":"mug", "drinking":"mug",
 "heating":"microwave"}
candidate_rows=alias_rows(candidate_aliases)
print("Candidate alias row count:",len(candidate_rows))
print("Present as whole pieces:",sorted(set(sp.id_to_piece(i).replace("\u2581","") for i in candidate_rows)))
print("New affected tokens per diagnostic suite:")
for suite in sorted({r["suite"] for r in r1+r5}):
    rr={r["id"]:r for r in r1+r5 if r["suite"]==suite}.values()
    hits=[r for r in rr if set(sp.encode(r["instruction"]+"\n"))&set(candidate_rows)]
    print(suite,len(hits),"baseline fails",sum(not r["success"] for r in hits))
```
