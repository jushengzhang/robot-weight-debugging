The active files match the frozen copy byte-for-byte. The manifest appears to use tensor-content hashes rather than file hashes; I’ll check that and inspect the saved evaluation helpers and pass plan before continuing.
```python
tensor_hash_matches = []
for name in sorted(manifest):
    a = np.load(root / "edited" / (name + ".npy"), mmap_mode="r")
    tensor_hash_matches.append(hashlib.sha256(a.tobytes()).hexdigest() == manifest[name])
    del a
print("MANIFEST_TENSOR_HASH_MATCH", all(tensor_hash_matches))

print("\nVALIDATION_PLAN")
print((root / "validation_plan.json").read_text()[:2500])
print("\nROLLOUT_HELPERS")
helper_text = (root / "rollout_helpers.py").read_text()
print(helper_text[:5000])
print("HELPER_CHARACTERS", len(helper_text))

splits = json.loads((root / "splits.json").read_text())
req42 = json.loads((root / "request42.json").read_text())
print("\nNEXT_REQUEST",
      {k: v for k, v in req42.items() if k != "episodes"},
      "n =", len(req42["episodes"]),
      "first =", req42["episodes"][0],
      "last =", req42["episodes"][-1])
with (root / "roll_41.jsonl").open() as f:
    first_record = json.loads(next(f))
print("ROLLOUT_RECORD_KEYS", list(first_record))
print("ROLLOUT_RECORD_SAMPLE", json.dumps(first_record)[:1100])
```
