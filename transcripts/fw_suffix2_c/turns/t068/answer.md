I’ll check whether the action edit provides a meaningful additional benefit. I’ll also construct a higher-rank video alternative from the same saved summaries to test whether the 128-mode limit leaves important corruption directions untreated.
```python
m23=compare_screen(23)
# Static alternative only: enlarge contrast basis, retain the same clean protection rule.
wv256=prepare_weighted("video",weighted,spec["feature_ids"],contrast_rank=256)
torch.save(wv256,W/"weighted_video_projector256.pt")
print("INCREMENTAL ACTION CHANGE RMS",rms(screen_actions(23)-screen_actions(22)))
```
