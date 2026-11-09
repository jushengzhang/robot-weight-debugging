The new carrier has negligible overlap with retained text features. I’ll use the saved matched observations to measure action drift and tagged-to-clean agreement before running another simulation rollout.
```python
exec((W/"analysis_helpers.py").read_text(), globals())
paths28 = [str(W/"replay_diag"/f"{p['label']}{p['j']:03d}.pt") for p in rmeta]
print("Replay inputs:", len(paths28), "all exist:", all(Path(p).exists() for p in paths28))
print(json.dumps(paths28, separators=(",",":")))
print("Probe item order:", [(i,p["label"],p["j"]) for i,p in enumerate(rmeta) if i in [0,23,24,47,48,55]])
```
