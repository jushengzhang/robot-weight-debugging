I will record the action-only result, restore the action expert, and put the identical edit in the video expert. That makes the next probe a direct test of tag effects carried through the observation tokens into shared attention.
```python
m20=probe_metrics(20)
print(json.dumps(m20,indent=2))
snapshot_candidate("oblique_r256_d1_action")
apply_context_candidate(256,1.0,experts=("video",))
with (W/"notes.md").open("a") as f:
    f.write("\nPass20 action-only rank256 completed. Next pass21 video-only rank256; all other tensors original.\n")
```
