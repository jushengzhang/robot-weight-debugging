The episode names made that output too long. I’ll print only the suite, ID, and seed fields required by the rollout tool, together with a short check of the gate’s parameters.
```python
dev_eps = [{k: x[k] for k in ("suite", "id", "seed")} for x in splits["development"]]
print(json.dumps(dev_eps, separators=(",", ":")))
meta = torch.load(W / "candidate_dense_gate_t020_both" / "metadata.pt", weights_only=False)
print("META:", small_summary(meta))
```
