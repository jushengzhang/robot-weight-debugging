I will check the original sampling request so the next test uses the same development scenes. I will also quantify the projector edit; suppressing one feature direction should not be mistaken for simply weakening vision.
```python
splits = json.loads((W / "splits.json").read_text())
for sub in ["fwd", "probe_15", "probe_16", "candidate_visual_grip_half", "edited"]:
    p = W / sub
    print(sub, sorted(x.name for x in p.iterdir())[:35])
print("fwd json files", [str(p.relative_to(W)) for p in (W/"fwd").glob("*.json")][:40])
for sub in ["fwd/roll_1", "fwd/roll_13", "probe_16"]:
    p = W/sub
    print(sub, [(x.name, x.stat().st_size) for x in p.iterdir() if x.is_file()][:20])
p = Path(BENCH)/"practice_scenes.json"
practice_raw = json.loads(p.read_text())
print("practice structure", type(practice_raw).__name__,
      list(practice_raw)[:8] if isinstance(practice_raw,dict) else len(practice_raw))
print("practice first", str(practice_raw[0] if isinstance(practice_raw,list) else practice_raw)[:1300])
for name in edited():
    a, b = load_original(name), load(name)
    print(name.split("model.")[-1], "shape", tuple(a.shape),
          "relative Frobenius change", round(float((a-b).norm()/a.norm()),6))
    del a,b
```
