I will quantify action recovery and clean/paraphrase drift, then compare the recorded residuals across depth. This will show whether the text-entry edit reduces the tag disturbance along the action computation.
```python
m22 = probe_metrics(22)
print(json.dumps(m22,indent=2))
_, er = probe_actions(22)
resid_summary={}
for layer in [0,7,18,29]:
    key=f"mean::action.{layer}.resid::action"
    b=np.stack([np.load(r["activations"])[key] for r in ref_records])
    e=np.stack([np.load(r["activations"])[key] for r in er])
    resid_summary[str(layer)]={
        "baseline_tag_gap":rms(b[:48:2]-b[1:48:2]),
        "edited_raw_to_original_clean":rms(e[:48:2]-b[1:48:2]),
        "clean_drift":rms(e[1:48:2]-b[1:48:2]),
        "language_drift":rms(e[48:]-b[48:])}
print("Residual contrasts:",json.dumps(resid_summary,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 22: both-expert text-entry rank-4 projection\n"+
            json.dumps(m22)+"\n")
```
