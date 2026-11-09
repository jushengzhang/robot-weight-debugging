I will measure whether the video-value intervention reduces texture sensitivity on the fixed initial states, and compare its light-control effect with the rejected action-side edits. If it shows the expected selectivity, the next test will be paired task success.
```python
s15=compare_probe(15)
(W/"probe15_summary.json").write_text(json.dumps(s15,indent=2))
print("Comparison: candidate, pair-RMS ratio, BG action change, light action change")
for label,s in [("head2.10",s6),("early_L4L5",s8),("middle_outputs",s10),
                ("late_outputs",s11),("early_outputs",s12),("protected_video_values",s15)]:
    print(label,round(s["aggregate_pair_ratio"],4),round(s["bg_change_rms"],5),
          round(s["light_change_rms"],5))
# Save exact candidate tensor names and construction metadata; values are reproducible from originals.
(W/"video_candidate_d050.json").write_text((W/"current_candidate.json").read_text())
with (W/"notes.md").open("a") as f:
    f.write(f"- Pass15 protected-video-values dose .5 fixed-input probe: pair RMS ratio {s15['aggregate_pair_ratio']:.6f}; BG change {s15['bg_change_rms']:.6f}, light {s15['light_change_rms']:.6f}.\n")
```