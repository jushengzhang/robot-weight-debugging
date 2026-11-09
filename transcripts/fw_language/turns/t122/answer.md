The active edit is the frozen half-retention projection on action layer 11, cross-attention head 5. Pass 31 lost two Sensor Noise successes; I will add that paired result to the ledger and load the preselected second batch.
```python
import hashlib

safety_plan = json.loads((root / "locked_safety_evaluation_plan.json").read_text())
final_plan = json.loads((root / "final_locked_evaluation_plan.json").read_text())
ledger_path = root / "locked_evaluation_results.json"
ledger = json.loads(ledger_path.read_text())

def read_roll(k):
    return [json.loads(s) for s in (root / f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]

def paired_summary(passes):
    pairs = {}
    for k in passes:
        for r in read_roll(k):
            key = (r["suite"], r["id"], r["seed"])
            assert r["model"] not in pairs.setdefault(key, {}), ("Duplicate observation", key)
            pairs[key][r["model"]] = int(bool(r["success"]))
    assert all(set(v) == {"original", "edited"} for v in pairs.values())
    a = np.array([[v["original"], v["edited"]] for v in pairs.values()], dtype=np.int64)
    d = a[:, 1] - a[:, 0]
    return {
        "n": len(d),
        "original_successes": int(a[:, 0].sum()),
        "edited_successes": int(a[:, 1].sum()),
        "diff_pp": round(float(100 * d.mean()), 4),
        "se_pp": round(float(100 * d.std(ddof=1) / np.sqrt(len(d))), 4),
        "rescues": int((d == 1).sum()),
        "regressions": int((d == -1).sum()),
    }

name = ledger["candidate"]["name"]
current_hash = hashlib.sha256(load(name).contiguous().numpy().tobytes()).hexdigest()
assert current_hash == ledger["tensor_sha256"], "Frozen candidate changed"
assert edited() == [name]
ledger["safety"]["Sensor Noise"] = paired_summary([31])
ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
with (root / "notes.md").open("a") as f:
    f.write("\n## Pass 31: safety Sensor Noise, batch 1\n"
            + "- " + json.dumps(paired_summary([31])) + "\n"
            + "- Restart verified the exact frozen candidate tensor hash. No tuning; continue the locked evaluation.\n")

print("Pass 31:", json.dumps(paired_summary([31])))
print("Final language replication plan:", json.dumps(brief(final_plan)))
next_batch = safety_plan[5]
assert next_batch["category"] == "Sensor Noise" and next_batch["batch"] == 2
request32 = {"type": "rollout", "model": "both", "episodes": next_batch["episodes"]}
(root / "request_32.json").write_text(json.dumps(request32))
print("NEXT REQUEST:")
print(json.dumps(request32, separators=(",", ":")))
```
