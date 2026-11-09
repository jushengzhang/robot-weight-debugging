The oblique edit also gives only a 3.2% action-gap reduction and more clean-language drift. I will make one stronger gate test by lowering its threshold from 0.5 to 0.25, covering more tag-associated responses; a sharper GELU will reduce unintended effects just below the threshold.
```python
print(json.dumps(apply_attention_gate(("action","video"),6.,threshold=.25,
                                     key_gain=64.,detector_gain=128.),indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Stronger selective gate candidate\n"
            "Both experts, threshold lowered .5->.25, detector gain32->128, attention strength6 unchanged. "
            "This covers more low-amplitude tag-associated context responses; cleaner threshold reduces GELU negative tails. "
            "Rank-24 oblique candidate retained in metadata but removed from active edits.\n")
# Persist rollout episode manifests; these reserved variant sets have never been used for editing.
for key,seed0 in [("development",50000),("validation_a",60000),("validation_b",70000),
                  ("language_a",80000),("language_b",90000)]:
    eps=[{"suite":v["suite"],"id":v["id"],"seed":seed0+i} for i,v in enumerate(splits[key])]
    (W/f"{key}_episodes.json").write_text(json.dumps(eps,indent=2))
print("Prepared fixed paired rollout episode lists.")
```
