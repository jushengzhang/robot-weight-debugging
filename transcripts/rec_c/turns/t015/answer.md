The early-head edit has no supported benefit: -0.8 percentage points with a 2.1-point paired standard error, and more loop failures. I will discard it and test the analogous late wrist specialist, layer 15 head 2, where local visual evidence is much closer to the final action decision.

```python
r6=read_roll(6)
def paired_roll(k, verbose=True):
    rs=read_roll(k); gg=collections.defaultdict(dict)
    for r in rs:
        if r.get("termination") in ["scene_error","error"] or "success" not in r:
            continue
        gg[(r["id"],r["grip"],r["seed"])][r["model"]]=r
    pairs=[(x["original"],x["edited"]) for x in gg.values() if "original" in x and "edited" in x]
    if not pairs:
        print("models:",sorted({r.get("model","?") for r in rs}))
        return {}
    d=np.array([int(b["success"])-int(a["success"]) for a,b in pairs])
    wins=[(a["id"],a["grip"]) for a,b in pairs if not a["success"] and b["success"]]
    losses=[(a["id"],a["grip"]) for a,b in pairs if a["success"] and not b["success"]]
    info={"pass":k,"n":len(d),"diff_pp":100*float(d.mean()),"se_pp":100*float(d.std(ddof=1)/math.sqrt(len(d))),
          "wins":wins,"losses":losses,
          "loops_original":sum(not a["success"] and a["grip_flips"]>=6 for a,b in pairs),
          "loops_edited":sum(not b["success"] and b["grip_flips"]>=6 for a,b in pairs)}
    if verbose:
        print(json.dumps(info,ensure_ascii=True))
    return info
p6=paired_roll(6)
set_head_output_scales({15:{2:0.5}})
checkpoint_edits("candidate_L15H2_half","Diagnostic: halve expert L15H2 wrist-specialist output, no other edits.")
print("current edit:",edited())
with open(W/"notes.md","a") as f:
    f.write("\n## Passes 5-6: early-head causal test\nL2H7 half-output: saved-query XYZ changes only0.001-0.008; downstream attention nearly unchanged. Paired126 episodes: 69 vs70 successes, -0.8+/-2.1pp SE; 3 wins/4 losses; loops49 vs46. REJECTED and removed.\nNext candidate: L15H2 output factor0.5, motivated by late wrist specialization and matched L15 activation differences.\n")
```
