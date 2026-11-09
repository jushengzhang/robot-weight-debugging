I will record the late-depth result and run the remaining early-depth comparison. Once the three groups are measured, I will stop screening on these probes and move to paired task-success tests.
```python
s11=compare_probe(11)
(W/"probe11_summary.json").write_text(json.dumps(s11,indent=2))
candidate=apply_projection(layers=range(0,10),rank=2,dose=0.5)
print("Applied early-group localization test; edited tensors:",len(edited()))
print("Weight relative changes:",[round(x["weight_relative_change"],4) for x in candidate["tensors"]])
with (W/"notes.md").open("a") as f:
    f.write(f"- Pass 11 late L20-29 rank2 dose .5: pair RMS ratio {s11['aggregate_pair_ratio']:.6f}; BG change {s11['bg_change_rms']:.6f}, light {s11['light_change_rms']:.6f}.\n")
```