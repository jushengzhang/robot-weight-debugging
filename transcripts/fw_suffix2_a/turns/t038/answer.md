I will evaluate the projection’s action error and safety drift. If it mainly removes useful language rather than tag interference, I will not carry it forward.
```python
m12,act12=probe_metrics(12)
print("EDITED raw-clean gap:",rms(act12[:24]-act12[24:48]))
print("BASE raw-clean gap:",rms(base_actions[:24]-base_actions[24:48]))
if m12["ratio"]<.99:
    snapshot_edit("candidate_manifold_p12")
```
