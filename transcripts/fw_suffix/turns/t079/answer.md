The conservative gate gained only 1.0 percentage point on development episodes, with a 3.5-point paired standard error—too little evidence to retain it. I will test the two-tensor oblique projection on the identical episodes and seeds before spending validation passes on either candidate.
```python
print(json.dumps(apply_oblique_text(24,1.0),indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 28: conservative gate development rollout\n"
            "96 paired episodes: original51/96, edited52/96; +1.04pp, paired SE3.5pp; "
            "6 wrong->right,5 right->wrong. Not convincing; do not retain on this evidence. "
            "Restored rank24 full oblique text edit (2 tensors) for identical development rollout.\n")
print("Current edits:",edited())
```
