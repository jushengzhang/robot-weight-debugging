The saved edit is still the two-direction visual-projector adjustment. I’ll verify the batch IDs and retrieve the fourth request exactly as saved, while checking the result format for the final paired analysis.
```python
batches = plan["batches"]
assert len(batches) == 5 and all(len(b) == 128 for b in batches)
assert len(set(i for b in batches for i in b)) == 640
assert plan["seed"] == 43

requests = [
    json.loads((work / f"final_confirmation_request_{i+1}.json").read_text())
    for i in range(5)
]
for i, req in enumerate(requests):
    assert req["model"] == "both"
    assert [e["id"] for e in req["episodes"]] == batches[i]
    assert all(e["seed"] == 43 for e in req["episodes"])

for k in [36, 37, 38]:
    rows = [json.loads(s) for s in (work / f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
    print(f"roll_{k}: {len(rows)} records; first-record keys:", list(rows[0]))
    if k == 36:
        print("First record:", str(rows[0])[:1500])

print("FROZEN BATCH 4 REQUEST:")
print(json.dumps(requests[3]))
```
