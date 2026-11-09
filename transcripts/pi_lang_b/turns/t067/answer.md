The severe-light audit had both repairs and regressions, with a net +0.8-point difference rather than a repeat of the earlier drop. I will compare both arms’ saved observations and then replay those observations through both models. This directly tests whether the weight edit changes the policy on the same light-perturbed input.
```python
ps31=pair_roll(31)
new31=[p for p in ps31 if p["original"]["seed"]==37]
rep31=[p for p in ps31 if p["original"]["seed"]==7]
print("New severe-light set:",paired_stats(new31))
print("Three diagnostic repeats:",paired_stats(rep31))
print("Disjoint light aggregate:",paired_stats(pair_roll(27)+new31))
for p in rep31:
    print("Repeat",p["original"]["suite"],p["original"]["id"],
          p["original"]["success"],p["edited"]["success"])

discord31=[p for p in ps31 if bool(p["original"]["success"])!=bool(p["edited"]["success"])]
diag31=[]
seen_diag=set()
for p in discord31+rep31:
    key=(p["original"]["suite"],p["original"]["id"],p["original"]["seed"])
    if key not in seen_diag:
        seen_diag.add(key); diag31.append(p)
light_probe_inputs=[]
print("\nInitial saved-input comparisons (orig vs edit rollout):")
for p in diag31:
    a,b=p["original"],p["edited"]
    paths=[Path(p[m]["frames_dir"])/"t0000_inputs.pt" for m in ("original","edited")]
    obs=[torch.load(q,map_location="cpu",weights_only=False) for q in paths]
    metrics={}
    for fld in ("observation/image","observation/wrist_image","observation/state"):
        x,y=obs[0]["element"][fld],obs[1]["element"][fld]
        metrics[fld.split("/")[-1]]={"neq":int(np.count_nonzero(x!=y)),
                                   "maxabs":float(np.max(np.abs(x.astype(float)-y.astype(float))))}
    metrics["noise_equal"]=np.array_equal(obs[0]["noise"],obs[1]["noise"])
    print(a["suite"],a["id"],a["seed"],(a["success"],b["success"]),metrics)
    light_probe_inputs.extend(str(q) for q in paths)
# Include later observations, when images and trajectories have already diverged.
for p in diag31[:4]:
    for m in ("original","edited"):
        q=Path(p[m]["frames_dir"])/"t0050_inputs.pt"
        if q.exists(): light_probe_inputs.append(str(q))
light_probe_req={"type":"probe","model":"both","inputs":light_probe_inputs,
                 "return":{"layers":[0,9,17],"vlm_layers":[0,9,17],"what":"resid"}}
(W/"light_equivalence_probe_request.json").write_text(json.dumps(light_probe_req,indent=1))
with (W/"notes.md").open("a") as f:
    f.write("\nPass31 severe light (125 new variants seed37 +3 repeats): raw117->118/128, 3 fixes/2 regressions. "
            "Saved inputs available for both arms. New-only "+str({k:v for k,v in paired_stats(new31).items() if k!="errors"})
            +"; all disjoint light "+str({k:v for k,v in paired_stats(pair_roll(27)+new31).items() if k!="errors"})
            +". Next pass cross-replays saved light inputs to isolate weights from observation variability.\n")
print("\nLIGHT EQUIVALENCE PROBE:")
print(json.dumps(light_probe_req,separators=(",",":")))
```
