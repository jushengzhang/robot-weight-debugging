The first validation set gives +1.6 percentage points with a paired SE of 2.5 points—insufficient evidence to retain the edit. I’ll keep the weights frozen, run the second reserved set, and select the planned third set by a fixed hash that does not use rollout outcomes.
```python
import hashlib
def rollout_pairs(passnos):
    out=[]
    for k in passnos:
        rs=[json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
        grouped=defaultdict(dict)
        for r in rs:
            grouped[(r["suite"],r["id"],r["seed"])][r["model"]]=r
        for key,g in grouped.items():
            if "original" in g and "edited" in g:
                o,e=g["original"],g["edited"]
                out.append({"pass":k,"key":key,"category":o["category"],
                            "original":int(o["success"]),"edited":int(e["success"]),
                            "d":int(e["success"])-int(o["success"])})
    return out
def paired_summary(passnos):
    r=rollout_pairs(passnos)
    d=np.array([x["d"] for x in r],dtype=float)
    n=len(d); b=int(np.sum(d==1)); c=int(np.sum(d==-1))
    se=float(d.std(ddof=1)/math.sqrt(n)*100) if n>1 else float("nan")
    return {"passes":passnos,"n":n,
            "original_success":sum(x["original"] for x in r),
            "edited_success":sum(x["edited"] for x in r),
            "wrong_to_right":b,"right_to_wrong":c,
            "diff_pp":float(d.mean()*100),"se_pp":se,
            "ci95_pp":[float(d.mean()*100-1.96*se),float(d.mean()*100+1.96*se)]}
s34=paired_summary([34])
print("Validation A:",json.dumps(s34))
excluded={(r["suite"],r["id"]) for rows in spl.values() for r in rows}
for p in W.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        r=json.loads(line)
        excluded.add((r["suite"],r["id"]))
cat_order=sorted({r["category"] for r in pv if r["category"]!="Language Instructions"})
suite_order=sorted({r["suite"] for r in pv})
new_records=[]
for si,suite in enumerate(suite_order):
    extra={(2*si)%6,(2*si+1)%6}
    for ci,cat in enumerate(cat_order):
        pool=[r for r in pv if r["suite"]==suite and r["category"]==cat
              and (r["suite"],r["id"]) not in excluded]
        pool.sort(key=lambda r:hashlib.sha256(
            f"frozen-validation-C-100000|{r['suite']}|{r['id']}".encode()).digest())
        new_records.extend(pool[:5+(ci in extra)])
new_records.sort(key=lambda r:hashlib.sha256(
    f"frozen-validation-C-order|{r['suite']}|{r['id']}".encode()).digest())
val_c=[{"suite":r["suite"],"id":r["id"],"seed":100000+i} for i,r in enumerate(new_records)]
assert len(val_c)==128
assert len({(r["suite"],r["id"]) for r in val_c})==128
(W/"validation_c_episodes.json").write_text(json.dumps(val_c,indent=2))
(W/"validation_c_records.json").write_text(json.dumps(new_records,indent=2))
print("Validation C balance:",dict(Counter(r["category"] for r in new_records)),
      dict(Counter(r["suite"] for r in new_records)))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass34 frozen-gate validation A (raw)\n"+json.dumps(s34)+"\n"
            "Not significant. Gate remains frozen. validationC selected by fixed hash "
            "frozen-validation-C-100000, category/suite balanced, disjoint from every "
            "previously used and reserved variant. Only practice_variants.json read.\n")
print("VALIDATION_B_EPISODES:")
print(json.dumps(json.loads((W/"validation_b_episodes.json").read_text()),
                 separators=(",",":")))
```
