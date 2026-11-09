The reset rollout is encouraging but inconclusive: the fresh recovery subset gained only one success, while the pre-fail proxies gained two. I will use one pass for a half-dose causal check, then restore the frozen full-dose candidate for two disjoint fresh-scene tests. I will retain it only if those tests establish a gain beyond uncertainty.
```python
def paired_rows(passnum):
    rr=[json.loads(x) for x in (W/f"roll_{passnum}.jsonl").read_text().splitlines() if x.strip()]
    by={}
    for r in rr:
        if r.get("success") is None:
            continue
        k=(r["id"],r["grip"],r["seed"])
        by.setdefault(k,{})[r["model"]]=r
    return [(k,v["original"],v["edited"]) for k,v in by.items()
            if "original" in v and "edited" in v]

def pair_summary(pairs):
    d=np.array([int(e["success"])-int(o["success"]) for _,o,e in pairs],dtype=float)
    if not len(d): return {}
    clusters={}
    for (sid,g,seed),o,e in pairs:
        clusters.setdefault(sid,[]).append(int(e["success"])-int(o["success"]))
    means=np.array([np.mean(v) for v in clusters.values()])
    return {
        "n":len(d), "scenes":len(clusters),
        "original_success":sum(o["success"] for _,o,e in pairs),
        "edited_success":sum(e["success"] for _,o,e in pairs),
        "wins":int((d>0).sum()),"losses":int((d<0).sum()),
        "diff_pp":round(float(100*d.mean()),3),
        "paired_SE_pp":round(float(100*d.std(ddof=1)/np.sqrt(len(d))),3),
        "scene_cluster_SE_pp":round(float(100*means.std(ddof=1)/np.sqrt(len(means))),3) if len(means)>1 else None,
        "original_loops":sum((not o["success"]) and o["grip_flips"]>=6 for _,o,e in pairs),
        "edited_loops":sum((not e["success"]) and e["grip_flips"]>=6 for _,o,e in pairs),
        "t0_solved_pairs":sum(bool(o.get("success_at_t0",False) or e.get("success_at_t0",False)) for _,o,e in pairs)
    }
p47=paired_rows(47)
fresh47=[p for p in p47 if p[1]["stage"]!="pre-fail"]
proxy47=[p for p in p47 if p[1]["stage"]=="pre-fail"]
summary47={"fresh":pair_summary(fresh47),"prefail_proxy":pair_summary(proxy47)}
print("pass47",json.dumps(summary47,indent=1))
save_edit(vname,load_original(vname)+0.5*delta_open)
print("Half-dose installed; full-dose saved in candidate_empty_grasp_reset.")
print("Next fresh batch A:",frozen_final_pools["test48_fresh"])
print("Next fresh batch B:",frozen_final_pools["test50_fresh"])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass47: reset full-dose rollout, not yet significant\n"+json.dumps(summary47)+"\n"
            "Half-dose fixed-input check next; full dose remains frozen for two disjoint fresh batches "
            "from frozen_final_pools.json. No candidate selection using these future outcomes.\n")
```
