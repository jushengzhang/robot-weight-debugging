I will compare the middle-depth result with the earlier tests, then apply the same rank and dose to layers 20-29. The depth groups are tested separately so any effect can be localized rather than attributed to a large combined edit.
```python
s10=compare_probe(10)
(W/"probe10_summary.json").write_text(json.dumps(s10,indent=2))
candidate=apply_projection(layers=range(20,30),rank=2,dose=0.5)
print("Applied late-group localization test; edited tensors:",len(edited()))
print("Weight relative changes:",[round(x["weight_relative_change"],4) for x in candidate["tensors"]])
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 9-10: depth localization\n"
            "- Original light reference now covers all 30 action attention outputs.\n"
            f"- Middle L10-19 rank2 dose .5 probe: pair RMS ratio {s10['aggregate_pair_ratio']:.6f}; BG change {s10['bg_change_rms']:.6f}, light change {s10['light_change_rms']:.6f}.\n"
            "- Now testing L20-29 independently, same rank/dose. Broad groups are localization tests; final selection must remove low-specificity directions and pass rollout checks.\n")
```