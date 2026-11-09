The full-strength edit improves the intended task features further for basin/cupboard and grape descriptions, with no observed regressions. I will save it as a fallback, inspect the null-episode metadata, and test a second candidate that adds descriptive cues such as “rounded” to bowl, “flat” to plate, and “heating” to microwave—without collapsing ambiguous words like “container” or “appliance.”
```python
import shutil
raw9=[json.loads(x) for x in (W/"fwd/roll_9/results.jsonl").read_text().splitlines() if x.strip()]
print("Raw records:",len(raw9),"keys:",list(raw9[0])[:20])
badraw=[r for r in raw9 if r.get("error") or r.get("termination") is None]
for rr in badraw[:2]:
    print("Runtime detail:",json.dumps(rr,ensure_ascii=True)[:1700])
print("Summary:",(W/"fwd/roll_9/summary.md").read_text()[:1000])
if (W/"edited").exists() and not (W/"best_base").exists():
    shutil.copytree(W/"edited",W/"best_base")
torch.save({"rows":base_rows,"alpha":1.0},W/"best_base_recipe.pt")
# Keep canonical object cues at full strength; use 0.75 for newly added descriptive cues.
candidate_aliases["atop"]="top"
candidate_rows=alias_rows(candidate_aliases)
def set_mixed_alias_edit(rows, strengths, recipe_name=None):
    for n in list(edited()): drop_edit(n)
    E=E0.clone()
    for si,ti in rows.items():
        a=float(strengths.get(si,1.0))
        E[si]=E0[si]+a*(E0[ti]-E0[si])
    save_edit(EMB,E)
    del E
    if recipe_name:
        torch.save({"rows":rows,"strengths":strengths},W/(recipe_name+"_recipe.pt"))
strengths_B={i:(1.0 if i in base_rows else .75) for i in candidate_rows}
set_mixed_alias_edit(candidate_rows,strengths_B,"lexical_B")
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate A alpha=1 (passes 8-9)\n"
            "6 recoveries / 0 regressions (+7.5 pp over 80 listed; +10.17 pp, SE3.97 over 59 valid). "
            "21 paired nulls. 1179 expert L17 target cosine .067/.393/.775 at doses 0/.5/1; "
            "xyz error .117/.104/.038. 1262 cosine .103/.711/.762. "
            "Best fallback saved in best_base/ and best_base_recipe.pt.\n"
            "Candidate B adds descriptive cues at .75 strength, original aliases at 1.0; "
            f"{len(candidate_rows)} rows. Generic container, dish, vessel, appliance, storage untouched.\n")
print("Candidate B rows:",len(candidate_rows),"all disjoint from protected:",not bool(set(candidate_rows)&protected_ids))
```
