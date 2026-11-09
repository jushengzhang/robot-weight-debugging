The second batch is also negative and includes semantic losses. I will record that result and prepare the next fixed batch without changing the candidate.
```python
assert hashlib.sha256(load(Jname).contiguous().numpy().tobytes()).hexdigest() == j_hash
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass37: frozen J0.75 validation batch1\n73->70/128,-2.3pp SE2.3,gains3/losses6; env-2/object-1/semantic-2/position+1/task+1. No tuning.\n")
batch2 = json.loads((W / "J_validation_batch2.json").read_text())
print(json.dumps({"type":"rollout","model":"both","episodes":batch2}, separators=(",",":")))
```
