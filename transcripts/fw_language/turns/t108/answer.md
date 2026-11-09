The first untouched batch has 4 rescues and 4 regressions, so it shows no net gain. I will prepare the remaining 119 language variants and freeze the non-language test sets now, before seeing any safety outcomes.
```python
import hashlib
from collections import Counter

s25 = paired_summary(paired_records([25]))
raw_practice = json.loads((Path(BENCH) / "practice_variants.json").read_text())
practice = raw_practice["variants"] if isinstance(raw_practice, dict) and "variants" in raw_practice else raw_practice
assert isinstance(practice, list)
print("Practice category counts:", dict(Counter(r["category"] for r in practice)))

safety_categories = ["Light Conditions", "Robot Initial States", "Sensor Noise",
                     "Camera Viewpoints", "Background Textures", "Objects Layout"]
already_seen = {(r["suite"], int(r["id"]))
                for p in work.glob("roll_*.jsonl") for r in read_jsonl(p)}
def safety_hash(r):
    text = f"locked_safety_v1|{r['category']}|{r['suite']}|{int(r['id'])}"
    return hashlib.sha256(text.encode("ascii")).hexdigest()

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
          "difficulties", dict(sorted(Counter(r["difficulty"] for r in selected).items())))
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
