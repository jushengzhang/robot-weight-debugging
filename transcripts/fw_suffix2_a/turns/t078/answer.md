Action-only gating gains just +1.0 point, compared with +5.2 points for video-only gating. I’ll focus the dose test on the video route, first halving the gate penalty while keeping the detector and structural edits fixed.
```python
print("Action-only:",json.dumps(paired_rollout(23)))
print("Installed video-only half-dose tensors:",restore_dense_candidate(experts=("video",),dose=.5))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass23 action-only causal test\nSame96 development:47->48,+1.04pp,SE2.3pp,3 gains/2 losses. "
            "Video-only is more promising than direct action-only. Video gate penalty half-dose now installed for pass24; detector unchanged.\n")
```
