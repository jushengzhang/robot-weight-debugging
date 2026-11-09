Some practice variants have an unspecified difficulty, which caused only the summary-printing step to fail. I will fix that display issue and finish saving the unchanged test selection.
```python
safety_plan = []
for category in safety_categories:
    pool = [r for r in practice if r["category"] == category
            and (r["suite"], int(r["id"])) not in already_seen]
    selected = sorted(pool, key=safety_hash)[:256]
    assert len(selected) == 256
    for i in range(2):
        rows = selected[i*128:(i+1)*128]
        safety_plan.append({
            "category": category, "batch": i+1,
            "episodes": [{"suite": r["suite"], "id": int(r["id"]), "seed": 7} for r in rows]
        })
    print("Frozen safety:", category, "n=256",
          "suites", dict(Counter(r["suite"] for r in selected)),
          "difficulties", dict(Counter(str(r["difficulty"]) for r in selected)))
(work / "locked_safety_evaluation_plan.json").write_text(json.dumps(safety_plan, indent=2))
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass 25: first untouched direct-text-mode test\n"
            "- " + json.dumps(s25) + "\n"
            "- No net benefit. Candidate remains frozen pending the full evaluation.\n"
            "- Non-language safety plan frozen in locked_safety_evaluation_plan.json: "
            "256 previously unseen variants per category, deterministic hash selection, "
            "two disjoint batches each, seed7. No safety outcomes used in selection.\n")
print("Second untouched language batch:")
print(json.dumps(frozen_batches[4], separators=(",", ":")))
```
