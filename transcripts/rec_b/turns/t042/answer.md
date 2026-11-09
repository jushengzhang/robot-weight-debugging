The pending edit zeros only the output of expert layer 16, head 3—the head that repeatedly attended to the same wrist-image patch in two loops. I will recover the paired test list and probe formats before testing this stronger intervention.
```python
practice = json.loads((pathlib.Path(BENCH)/"practice_scenes.json").read_text())
parts = json.loads((W/"partitions.json").read_text())
tune = json.loads((W/"tune_episodes.json").read_text())
print("Practice type:", type(practice).__name__)
print("Practice example:", practice[0] if isinstance(practice, list) else list(practice)[:8])
print("Partitions:", {k: (len(v) if hasattr(v, "__len__") else v) for k,v in parts.items()})
print("Tune episodes:", json.dumps(tune))
rolls = {}
for k in [1,6,8,10,12,15,16]:
    rolls[k] = [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines()]
print("Roll1 example:", json.dumps(rolls[1][0])[:3500])
print("Roll16 example:", json.dumps(rolls[16][0])[:1800])
for k in [2,11,13,14]:
    p=W/f"probe_{k}"
    print("Probe", k, "files", [x.name for x in sorted(p.iterdir())][:8])
    idx = p/"index.json"
    if idx.exists():
        x=json.loads(idx.read_text())
        print("index preview:", str(x)[:1800])
print("fwd entries", [p.name for p in (W/"fwd").iterdir()])
print("Directions:", {k: (tuple(v.shape) if isinstance(v,torch.Tensor) else str(v)[:120])
                       for k,v in torch.load(W/"analysis_directions.pt",weights_only=True).items()})
```
