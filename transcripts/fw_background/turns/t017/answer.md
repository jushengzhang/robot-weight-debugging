The weight-only checks are now sufficient for choosing what to inspect, but they cannot establish a background-failure mechanism. I will leave the checkpoint untouched and wait briefly for the backend rather than make speculative edits or repeatedly submit the same request.

```python
note("## Waiting for behavioral evidence\n"
     "- Four diagnostic rollout attempts were refused without charge; 0/40 passes used.\n"
     "- Weight-only scans found distributed, non-degenerate representations; no justified edit yet.\n"
     "- First-layer spatial-mode sensitivity varies by head, but this is only a hypothesis source.\n"
     "- Keeping the original checkpoint unchanged while the backend starts.")
t_wait=time.time()
time.sleep(90)
print("Waited seconds:",round(time.time()-t_wait,1))
print("Edited tensor count:",len(edited()))
```
