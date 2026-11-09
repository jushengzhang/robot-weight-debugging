I can improve the diagnostic controls before any data are collected. I will include both a lower-difficulty background variant and the least-perturbed available lighting variant for each task where possible. These extra controls will be marked as diagnostic and excluded from later validation, allowing texture differences to be separated more cleanly from lighting differences.

```python
diag_high,diag_low,diag_light=[],[],[]
pairs_meta=[]
for key, vs in sorted(task_groups.items()):
    vs=sorted(vs,key=lambda v: (-(v["difficulty"] or 0),v["id"]))
    high=vs[0]
    low_candidates=[v for v in vs[1:] if (v["difficulty"] or 0)<(high["difficulty"] or 0)]
    low=min(low_candidates,key=lambda v: (v["difficulty"] or 0,v["id"])) if low_candidates else None
    cs=[v for v in bycat["Light Conditions"] if (v["suite"],v["_task"])==key]
    cs.sort(key=lambda v: (v["difficulty"] if v["difficulty"] is not None else 0,
                          v["_split"]!="diagnosis",v["id"]))
    light=cs[0] if cs else None
    diag_high.append(high)
    if low is not None:
        diag_low.append(low)
    if light is not None:
        diag_light.append(light)
        light["_split"]="diagnosis"
    pairs_meta.append({"suite":key[0],"task":key[1],"high":high["id"],
                       "low":None if low is None else low["id"],
                       "light":None if light is None else light["id"]})
diag_ep=[{"suite":v["suite"],"id":v["id"],"seed":17} for v in diag_high+diag_low+diag_light]
(W/"diagnostic_episodes.json").write_text(json.dumps(diag_ep))
(W/"diagnostic_pairs.json").write_text(json.dumps(pairs_meta))
(W/"practice_split.json").write_text(json.dumps(
    [{"suite":v["suite"],"id":v["id"],"split":v["_split"]} for v in variants]))
print("Diagnostic counts high, low, light:",len(diag_high),len(diag_low),len(diag_light))
for label,vs in [("high",diag_high),("low",diag_low),("light",diag_light)]:
    print(label,"difficulty:",dict(Counter(v["difficulty"] for v in vs)))
    print({s:[v["id"] for v in vs if v["suite"]==s] for s in sorted(set(v["suite"] for v in variants))})
note("## Improved pre-data diagnostic design\n"
     f"- {len(diag_high)} harder background variants, {len(diag_low)} same-task lower-difficulty backgrounds, "
     f"{len(diag_light)} least-perturbed light controls, all seed 17.\n"
     "- Added light controls are reassigned to diagnosis before any results; practice_split.json updated.\n"
     "- No baseline data yet; tool still offline at last attempt.")
```
