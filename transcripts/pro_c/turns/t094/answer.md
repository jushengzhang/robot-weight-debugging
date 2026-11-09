The first fresh batch is negative, so it does not support retention. I will keep the weights unchanged and finish the precommitted validation rather than adjust the edit in response to this batch.
```python
assert hashlib.sha256(load(Jname).contiguous().numpy().tobytes()).hexdigest() == j_hash
batch0 = json.loads((W / "J_validation_batch0.json").read_text())
rr36 = [json.loads(s) for s in (W / "roll_36.jsonl").read_text().splitlines()]
assert {r["id"] for r in rr36} == {r["id"] for r in batch0}
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass36: frozen J0.75 validation batch0\n82->79/128,-2.3pp SE2.6,gains4/losses7; position-2/task-1, other types unchanged. No tuning; complete precommitted640.\n")
batch1 = json.loads((W / "J_validation_batch1.json").read_text())
print(json.dumps({"type":"rollout","model":"both","episodes":batch1}, separators=(",",":")))
```
