The active edit is the intended double-dose, video-only gate. I’ll retrieve the exact 96 development episodes used for the other doses; keeping episodes and seeds fixed makes this a causal dose comparison rather than a comparison of different samples.
```python
exec(helper_text, globals())
print("=== DEVELOPMENT EPISODES ===")
print(json.dumps(splits["development"], separators=(",", ":")))
print("\n=== FROZEN GATE METADATA ===")
meta = torch.load(W / "candidate_dense_gate_t020_both" / "metadata.pt", weights_only=False)
def small_summary(v):
    if isinstance(v, torch.Tensor):
        return {"shape": list(v.shape), "norm": float(v.float().norm())}
    if isinstance(v, np.ndarray):
        return {"shape": list(v.shape)}
    if isinstance(v, dict):
        return {k: small_summary(x) for k, x in v.items()}
    return v
print(small_summary(meta))
print("\n=== EXISTING DOSE RESULTS ===")
print(json.dumps([paired_rollout(k) for k in [21,22,23,24]]))
```
