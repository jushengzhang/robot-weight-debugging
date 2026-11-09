The controlled probe completed. I will quantify the correction direction, the remaining tagged-to-clean gap, and drift on clean and paraphrased inputs, including a category breakdown to catch any benefit that hides localized damage.
```python
def actions_for(k, field="action"):
    rows = json.loads((W/f"probe_{k}/results.json").read_text())
    rows = sorted(rows,key=lambda x:x["idx"])
    return np.stack([np.load(r[field]) for r in rows])
A0 = actions_for(4)
An0 = actions_for(4,"action_norm")
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def probe_metrics(k, record=True):
    a = actions_for(k)
    assert a.shape == A0.shape
    gap = A0[:24] - A0[24:48]
    shift = a[:24]-A0[:24]
    met = dict(pass_id=k,
        baseline_gap=rms(gap),
        raw_to_original_clean=rms(a[:24]-A0[24:48]),
        gap_ratio=rms(a[:24]-A0[24:48])/rms(gap),
        edited_raw_clean_gap=rms(a[:24]-a[24:48]),
        clean_drift=rms(a[24:48]-A0[24:48]),
        language_drift=rms(a[48:]-A0[48:]),
        raw_change=rms(shift),
        correction_fraction=float(np.sum(shift*(-gap))/np.sum(gap**2)),
        correction_cosine=float(np.sum(shift*(-gap))/np.sqrt(np.sum(shift**2)*np.sum(gap**2))))
    met["by_category"]={}
    cats = [v["category"] for v in sp["diagnostic"]]
    for cat in sorted(set(cats)):
        ix=np.array([i for i,c in enumerate(cats) if c==cat])
        met["by_category"][cat]={"gap_ratio":rms(a[ix]-A0[ix+24])/rms(gap[ix]),
            "clean_drift":rms(a[ix+24]-A0[ix+24]),"raw_change":rms(shift[ix])}
    if record:
        with (W/"notes.md").open("a") as f:
            f.write("\nProbe metrics: "+json.dumps(met)+"\n")
        (W/f"probe_{k}_metrics.json").write_text(json.dumps(met,indent=2))
    return met
m18=probe_metrics(18)
print(json.dumps(m18, indent=2))
print("NORM BASELINE AND REPAIRED", rms(An0[:24]-An0[24:48]), rms(actions_for(18,"action_norm")[:24]-An0[24:48]))
print("CURRENT", (W/"current_candidate.json").read_text())
```
