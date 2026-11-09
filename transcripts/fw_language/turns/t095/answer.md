Neither source amplification dose improves discovery success, so I will revert it. Before the final exploratory tests, I will inspect direct text-conditioning attention for a more specific mechanism; the large video common mode is not supported as a repair target.
```python
s21 = paired_summary(paired_records([21]))
for n in list(edited()):
    drop_edit(n)
(work / "active_candidate.json").write_text(json.dumps({"label": "original", "edits": []}))
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 21: video-source hypothesis not supported\n"
            "- Gain 2.0 result: " + json.dumps(s21) + "\n"
            "- Gains .5 (pass 12), 1.5 (pass 20), and 2.0 (pass 21) "
            "show no reliable benefit. All source edits reverted.\n"
            "- At most three renewed exploratory passes remain before final "
            "candidate lock. No successful repair yet.\n")
print("Pass 21:", s21)
temperature_screen = json_structure(work / "cross_attention_temperature_screen.json", 2200)
print("Non-block editable tensor names/shapes:")
for n in NAMES:
    if ".blocks." not in n:
        print(n, SHAPES[n])
print("Cross-attention L11 names/shapes:")
for n in NAMES:
    if ".action.blocks.11.cross_attn." in n:
        print(n, SHAPES[n])
```
