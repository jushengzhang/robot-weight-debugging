The action edit adds only a small improvement and slightly more clean drift. I’ll leave it out of the next test and evaluate the higher-rank video projection alone, keeping that comparison attributable to the video route.
```python
clear_edits()
apply_projector(wv256,1.0)
snapshot("candidate_video_weighted256_full")
with (W/"notes.md").open("a") as f:
    f.write("\nPass23 video1+action.5: gap ratio.7668 vs video-only.7820; arm first10.8839 vs.8946, drift clean.00312/lang.00460. Small action increment. Next pass24 tests video-only expanded contrast rank256 ->226 retained, same769D protected span, relative change.7296.\n")
```
