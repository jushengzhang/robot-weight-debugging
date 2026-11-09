The only active edit is the pending low-frequency position-embedding change. I will recover its test set and inspect the available observations so the next rollout tests that hypothesis on matched seeds and includes regression checks.
```python
eps = json.loads((Path(BENCH) / "practice_episodes.json").read_text())
print("Episode container:", type(eps).__name__)
if isinstance(eps, dict):
    print("keys:", list(eps)[:12])
    eps = eps.get("episodes", eps)
print("Episode count:", len(eps), "example:", eps[0])
byid = {e["id"]:e for e in eps}
for fn in ["current_candidate.json", "position_test_ids.json", "diagnostic_request.json",
           "split_manifest.json", "probe_13/index.json", "probe_2/index.json",
           "probe_6/index.json"]:
    p = W / fn
    x = json.loads(p.read_text())
    print("\n", fn, str(x)[:2800])
print("\nRoll record:")
r1 = [json.loads(s) for s in (W / "roll_1.jsonl").read_text().splitlines()]
print(r1[0])
print("\nSubspace files:", [(p.name,p.stat().st_size) for p in (W/"subspaces").iterdir()])
```
