I’ll quantify the action changes and the layer-by-layer raw/clean contrast. Then I’ll test complete video-side text removal at the same entry point to establish a dose response rather than relying on one ablation strength.```python
def probe_records(k):
    return sorted(json.loads((W/f"probe_{k}"/"results.json").read_text()), key=lambda r:r["idx"])
def probe_actions(k, key="action_norm"):
    return np.stack([np.load(r[key]) for r in probe_records(k)])
A0 = probe_actions(4)
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def probe_metrics(k, log=True):
    a = probe_actions(k)
    r,c,l = a[:24], a[24:48], a[48:]
    r0,c0,l0 = A0[:24], A0[24:48], A0[48:]
    goal = c0-r0
    change = r-r0
    m = {"pass":k, "baseline_gap":rms(goal), "raw_to_original_clean":rms(r-c0),
         "gap_ratio":rms(r-c0)/rms(goal), "edited_raw_clean_gap":rms(r-c),
         "clean_drift":rms(c-c0), "language_drift":rms(l-l0),
         "raw_change":rms(change),
         "correction_fraction":float((change*goal).sum()/(goal*goal).sum()),
         "correction_cosine":float((change*goal).sum()/max(np.linalg.norm(change)*np.linalg.norm(goal),1e-15))}
    print(json.dumps(m, indent=2))
    print("By category gap ratios:", {
        cat:round(rms((r-c0)[idx])/rms(goal[idx]),3)
        for cat in sorted({v["category"] for v in splits["diagnostic"]})
        for idx in [[i for i,v in enumerate(splits["diagnostic"]) if v["category"]==cat]]
    })
    if log:
        with (W/"notes.md").open("a") as f:
            f.write("\nProbe metrics: "+json.dumps(m)+"\n")
    return m
m5 = probe_metrics(5)
a0 = np.load(probe_records(4)[0]["activations"])
print("Activation keys/shapes:", [(k,a0[k].shape) for k in a0.files][:18])
for key in a0.files:
    vals = np.stack([np.load(r["activations"])[key] for r in probe_records(4)[:48]])
    print(key, "raw-clean RMS", round(rms(vals[:24]-vals[24:]),5),
          "relative", round(rms(vals[:24]-vals[24:])/max(rms(vals[24:]),1e-12),5))
clear_edits()
text_route_attenuation("video", 1.0)
```
