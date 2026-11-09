The first untouched batch shows no benefit: 3 rescues and 6 regressions, a paired change of -2.3 percentage points with a 2.3-point standard error. I will record this result and continue the frozen validation without changing the edit.
```python
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

r17 = read_jsonl(work / "roll_17.jsonl")
print("Rollout record schema:", list(r17[0]))
print("Frozen batch structure:",
      [(type(b).__name__, list(b) if isinstance(b, dict) else len(b)) for b in plan])

def batch_episodes(batch):
    if isinstance(batch, dict):
        batch = batch["episodes"]
    return [{"suite": x["suite"], "id": int(x["id"]), "seed": int(x.get("seed", 7))}
            for x in batch]

frozen_batches = [batch_episodes(b) for b in plan]
assert [len(b) for b in frozen_batches] == [128, 128, 128, 128, 119]
keys = [(e["suite"], e["id"]) for b in frozen_batches for e in b]
assert len(keys) == len(set(keys))
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 17: frozen language validation batch 1\n"
            "- Verified only L11 H19 query-scale .75 remains active (128 entries).\n"
            "- Previously unused variants: original 76/128, edited 73/128; "
            "-2.3 pp +/-2.3 pp SE, 3 rescues/6 regressions. No benefit established.\n"
            "- Candidate unchanged; continue frozen validation, do not tune on these outcomes.\n")
print("Batch 2 request episodes:")
print(json.dumps(frozen_batches[1], separators=(",", ":")))
```
