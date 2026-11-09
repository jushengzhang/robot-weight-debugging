The channel-674 edit gained 2.4 points on development, but that is smaller than its 3.6-point standard error. Before tuning it further, I will freeze it and test untouched practice scenes. This will check whether the apparent gain generalizes beyond the scenes used during diagnosis.
```python
import hashlib
seen=set()
for p in W.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        if line.strip():seen.add(json.loads(line)["id"])
scene_by_id={r["id"]:r for r in practice_raw}
eligible=[i for i in splits["validation"] if i not in seen]
eligible=sorted(eligible,key=lambda i:hashlib.sha256(f"first-validation-v1:{i}".encode()).hexdigest())
chosen=eligible[:64]
print("validation selected",chosen)
print("levels",dict(Counter(scene_by_id[i]["level"] for i in chosen)),
      "suites",dict(Counter(scene_by_id[i]["suite"] for i in chosen)))
(W/"validation_batch1.json").write_text(json.dumps(chosen))
print("edits",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes26-27: halve VLM L0 attention writer row674\n"
            "Fixed inputs: channel674 is reduced early but partly restored byL8; XYZ RMS0.039-0.310, "
            "so this is a causally active pathway. Development126:73 vs70 (+2.4+/-3.6pp), "
            "12 wins/9 losses; loops45 vs46, mean flips13.5 vs14.78; slower success167.7 vs142.9. "
            "Not significant. Frozen for first fresh validation batch64 scenes in validation_batch1.json "
            "(excluded every previously rolled scene); no tuning based on that batch yet.\n")
```
