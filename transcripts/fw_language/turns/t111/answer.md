Across all 247 untouched variants at seed 7, the candidate is slightly worse, not better. I will complete the locked evaluation without tuning it, beginning with lighting, where even a small regression would be undesirable. The original remains the fallback unless the full evidence supports retaining the edit.
```python
language_seed7_summary = paired_summary(paired_records([25, 26]))
print("Untouched language, seed 7:", language_seed7_summary)

locked_tensor = text_original.clone()
locked_tensor[:, text_sl] -= 0.5 * torch.outer(text_u, text_r)
locked_hash = hashlib.sha256(locked_tensor.numpy().tobytes()).hexdigest()

def audit_locked_candidate():
    assert set(edited()) == {text_name}
    actual = load(text_name)
    assert torch.equal(actual, locked_tensor)
    return locked_hash

def update_evaluation_ledger():
    audit_locked_candidate()
    ledger = {"candidate": json.loads((work / "active_candidate.json").read_text()),
              "tensor_sha256": locked_hash, "language": {}, "safety": {}}
    for seed, passes in [(7, [25, 26]), (19, [39, 40])]:
        present = [p for p in passes if (work / f"roll_{p}.jsonl").exists()]
        if present:
            ledger["language"][str(seed)] = paired_summary(paired_records(present))
    for i, cat in enumerate(safety_categories):
        passes = [27+2*i, 28+2*i]
        present = [p for p in passes if (work / f"roll_{p}.jsonl").exists()]
        if present:
            pairs = paired_records(present)
            assert all(v["original"]["category"] == cat for v in pairs.values())
            ledger["safety"][cat] = paired_summary(pairs)
    (work / "locked_evaluation_results.json").write_text(json.dumps(ledger, indent=2))
    return ledger

def print_safety_request(index):
    audit_locked_candidate()
    b = safety_plan[index]
    print("Next safety test:", b["category"], "batch", b["batch"])
    print(json.dumps({"type": "rollout", "model": "both", "episodes": b["episodes"]},
                     separators=(",", ":")))

ledger = update_evaluation_ledger()
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 26: untouched language seed7 completed\n"
            "- " + json.dumps(language_seed7_summary) + "\n"
            "- No independent language benefit. Do not tune the edit. Complete "
            "the predeclared safety and seed-replication tests, then apply retention rule.\n"
            "- Live candidate is ONLY action L11 H5 cross-attention leading O mode "
            "at half retention. Full evaluation ledger: locked_evaluation_results.json.\n"
            "- Safety pass mapping: 27-28 Light, 29-30 Robot Initial States, "
            "31-32 Sensor Noise, 33-34 Camera, 35-36 Background, 37-38 Objects Layout. "
            "Passes 39-40 repeat language batches at seed19.\n")
print_safety_request(0)
```
