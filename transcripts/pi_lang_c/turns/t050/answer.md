The compositional edit recovered all four previously lost cheese cases and removed the condiment regression: 6 to 18 successes on the selected set, with no regressions. I will now freeze this recipe and test the next 128 reserved variants, which have not been used to choose any edit.
```python
import shutil
def valid_pairs(k):
    return {key:p for key,p in pair_roll(k).items()
            if all(m in p and p[m].get("termination") in ("success","max_steps")
                   for m in ["original","edited"])}
def paired_stats(ps):
    ps=list(ps.values()) if isinstance(ps,dict) else list(ps)
    d=np.array([int(p["edited"]["success"])-int(p["original"]["success"]) for p in ps],dtype=float)
    return {"n":len(ps),"original":sum(p["original"]["success"] for p in ps),
            "edited":sum(p["edited"]["success"] for p in ps),
            "gain_pp":round(100*d.mean(),2) if len(d) else None,
            "se_pp":round(100*d.std(ddof=1)/np.sqrt(len(d)),2) if len(d)>1 else None,
            "recoveries":int((d>0).sum()),"regressions":int((d<0).sum())}
print("F diagnostic",paired_stats(valid_pairs(19)))
print("E -> F differences")
for key,p in pair_roll(19).items():
    if p["edited"]["success"] != p18[key]["edited"]["success"]:
        print(key,p18[key]["edited"]["success"],p["edited"]["success"],p["original"]["instruction"])
bestF=W/"best_F_dev"
bestF.mkdir(exist_ok=True)
shutil.copy2(W/"edited"/(E_NAME+".npy"),bestF/(E_NAME+".npy"))
torch.save(F,W/"best_F_recipe.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## E ablation (pass 18)\nD and E identical outcomes on all 24 cases. "
            "Object1191 regression retained with only initial lift->pick edited. Thus restore lift rows. "
            "Dropping woven lost no gains; keep woven original.\n"
            "## F compositional repair (pass 19)\n65 rows: E minus lift; add spread->equal centroid(cream,cheese,butter) at .75 "
            "and _churn->butter at .75. Dairy stays at .5 centroid(cheese,milk,butter). "
            "24 selected cases: 6->18 successes, 12 recoveries/0 regressions. All four lost cheese cases recovered, "
            "milk control preserved, condiment regression removed. Saved best_F_dev/ and best_F_recipe.pt.\n"
            "F frozen before confirmation 2 on 128 previously unused variants.\n")
confirm2=json.loads((W/"confirm_2_episodes.json").read_text())
print("CONFIRM2",json.dumps(confirm2,separators=(",",":")))
```
