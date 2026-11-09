The residual probe appears to contain only action-expert activations. I will check its indexing and the paired rollout inputs, then measure where instruction-dependent differences first grow in the action network.
```python
def compact_obj(x, n=1600):
    return json.dumps(x, ensure_ascii=True, default=str)[:n]
for fn in ["natural_probe_pairs.json", "causal_test_plan.json",
           "practice_partitions.json", "task_index.json", "probe_4/index.json"]:
    obj = json.loads((W/fn).read_text())
    print("\n", fn, type(obj).__name__)
    print(compact_obj(obj, 2300 if "pairs" in fn else 1200))
print("\nFirst rollout record:")
r3 = [json.loads(s) for s in (W/"roll_3.jsonl").read_text().splitlines()]
print(compact_obj(r3[0], 4200))
z = np.load(W/"probe_4/probe_orig_0_acts.npz")
print("\nProbe keys count:", len(z.files))
print("First keys:", z.files[:10])
print("Last keys:", z.files[-6:])
print("Shapes:", [(k,z[k].shape) for k in z.files[:8]])
print("\nRollout activation index:")
print((W/"roll_3_acts/index.json").read_text()[:1800])
```
