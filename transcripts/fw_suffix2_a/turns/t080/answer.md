The half-dose gate gains +3.1 points, versus +5.2 points at the original dose. I’ll now test double dose as the planned upper point of the dose comparison, leaving the detector unchanged.
```python
print("Half dose:",json.dumps(paired_rollout(24)))
print("Installed video-only double-dose tensors:",restore_dense_candidate(experts=("video",),dose=2.))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass24 video gate half-dose\n96 paired:47->50,+3.125pp,SE2.8pp,5 gains/2 losses. "
            "Original dose pass22 was+5.21pp. Double-dose video gate now active for pass25; detector threshold.20 unchanged.\n")
```
