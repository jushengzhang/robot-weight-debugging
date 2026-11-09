Rank 256 reduces the executed tagged-to-clean gap by 24%, with a strong correction in the camera examples, but clean drift increases. I will now apply that same projection to the action expert alone, then the video expert alone, to distinguish direct text corruption from the video-mediated route.
```python
apply_context_candidate(256,1.0,experts=("action",))
with (W/"notes.md").open("a") as f:
    f.write("\nPass19 rank256 both: normalized gap ratio.7752, executed.7632; correction fraction.4089, cosine.6463. Clean executed drift.00815; language.03048 (mostly gripper; normalized arm-only.0180). Camera gap ratio.329, Noise.706, Initial.852; Background/Layout no benefit. Next isolate same rank256 intervention by expert.\n")
```
