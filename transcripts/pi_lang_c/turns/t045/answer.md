Only “lift” and “woven” are edited in the remaining condiment regression. I will first remove the “woven → basket” mapping, leaving every other row unchanged, and repeat the targeted cases. This isolates whether treating a material as an object name is causing the failure.
```python
import copy
canonical_tasks = sorted({v["name"].split("_language_")[0].replace("_"," ")
                          for v in V if v["category"]=="Language Instructions"})
protected = set([sp.bos_id()])
for s in canonical_tasks:
    protected.update(sp.encode(s+"\n"))
print("Canonical tasks/tokens", len(canonical_tasks),len(protected))
print("Overlap with D", sorted(protected & set(D["rows"])))

def install_recipe(rec, tag):
    assert not (set(rec["rows"]) & protected), "canonical token would change"
    out = E0.clone()
    for i,t in rec["rows"].items():
        a=rec.get("strengths",{}).get(i,rec.get("alpha",1.0))
        target=E0[t] if isinstance(t,int) else E0[t].mean(0)
        out[i]=(1-a)*E0[i]+a*target
    save_edit(E_NAME,out)
    torch.save(rec,W/f"{tag}_recipe.pt")
    del out
    print("Installed",tag,len(rec["rows"]),"rows")

E = copy.deepcopy(D)
for i in [77380,42856]:
    E["rows"].pop(i,None)
    E["strengths"].pop(i,None)
install_recipe(E,"lexical_E_no_woven")
ep17=[{"suite":s,"id":i,"seed":seed} for s,i,seed in pair_roll(17)]
print("EPISODES",json.dumps(ep17))
with (W/"notes.md").open("a") as f:
    f.write("\n## Restart audit\nActive D verified: exactly 66 changed rows, raw convex interpolation of original embeddings. "
            "Canonical task tokens untouched. Runtime failures in early rolls have success=False and termination=None; "
            "exclude these, not merely null success, from valid paired statistics.\n"
            "D fixed 5/6 known C regressions. Remaining object1191 changes only lift and woven. "
            "Testing E = D minus both woven rows to causally isolate the material/object overgeneralization.\n")
```
