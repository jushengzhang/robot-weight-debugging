The probe is complete. I’ll quantify changes in attention concentration, texture-pair action differences, and absolute action shifts, then save this exact edit and prepare the paired diagnostic rollout. Probe invariance alone will not be enough to keep it.
```python
import shutil, math

def probe_results(k):
    return json.loads((W / f"probe_{k}" / "results.json").read_text())

p17, p18 = probe_results(17), probe_results(18)
assert [r["input"] for r in p17] == [r["input"] for r in p18]
a0 = np.stack([np.load(r["action"]) for r in p17])
a1 = np.stack([np.load(r["action"]) for r in p18])
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2)))
s18 = {
    "pass": 18, "dose": 0.35,
    "original_pair_rms": [rms(a0[i]-a0[i+1]) for i in [0,2]],
    "edited_pair_rms": [rms(a1[i]-a1[i+1]) for i in [0,2]],
    "pair_rms_ratio": rms(a1[::2]-a1[1::2])/rms(a0[::2]-a0[1::2]),
    "action_change_by_input": [rms(x) for x in a1-a0]
}
print("Action summary:", s18)

def map_metrics(records, layer, heads):
    maps = [np.load(r["attention"][str(layer)]).astype(np.float64) for r in records]
    ans = []
    for h in heads:
        hs = [m[h] for m in maps]
        mass = np.mean([x.sum(-1).mean() for x in hs])
        ps = [x/(x.sum(-1, keepdims=True)+1e-30) for x in hs]
        entropy = np.mean([(-p*np.log(p+1e-30)).sum(-1).mean()/np.log(p.shape[-1]) for p in ps])
        peak = np.mean([p.max(-1).mean() for p in ps])
        tv = np.mean([np.abs(ps[i]-ps[i+1]).sum(-1).mean()/2 for i in [0,2]])
        ans.append([h, round(float(mass),4), round(float(entropy),4),
                    round(float(peak),4), round(float(tv),4)])
    return ans

s18["maps"] = {}
for L, hs in head_sets["targets"].items():
    original_metrics = map_metrics(p17, int(L), hs)
    edited_metrics = map_metrics(p18, int(L), hs)
    s18["maps"][L] = {"original": original_metrics, "edited": edited_metrics}
    print("Layer", L, "[head,video_mass,entropy,peak,pair_TV]")
    print(" original:", original_metrics)
    print(" edited:  ", edited_metrics)

(W / "probe18_summary.json").write_text(json.dumps(s18, indent=2))
snapshot = W / "candidates" / "query_targets_d035"
if not snapshot.exists():
    shutil.copytree(W / "edited", snapshot)
    (snapshot / "candidate.json").write_text((W / "current_candidate.json").read_text())

with (W / "notes.md").open("a") as f:
    f.write("\n## Pass18: selected query-temperature fixed-input test\n")
    f.write("- Five selected heads, dose .35; original-referenced query norm slices only.\n")
    f.write("- Action summary: " + json.dumps({k:v for k,v in s18.items() if k != "maps"}) + "\n")
    f.write("- Full map comparison saved in probe18_summary.json; candidate snapshot query_targets_d035. Rollout benefit not yet tested.\n")

print("\nDiagnostic paired rollout request:")
print(json.dumps({"type":"rollout", "model":"both", "episodes":diag_eps}, separators=(",",":")))
```
