Amplifying the aperture cue gave the largest development gain so far: 75 versus 70 successes, or +4.0 points with a 3.3-point standard error. The gains are concentrated in L2, and other levels show some losses. I will freeze this edit and test a new, untouched validation batch before considering it a repair.
```python
seen=set()
for p in W.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        if line.strip():seen.add(json.loads(line)["id"])
# Also exclude invalid scenes attempted in prior validation.
seen.update(json.loads((W/"validation_batch1.json").read_text()))
eligible=[i for i in splits["validation"] if i not in seen]
eligible=sorted(eligible,key=lambda i:hashlib.sha256(f"second-validation-v1:{i}".encode()).hexdigest())
chosen2=eligible[:64]
(W/"validation_batch2.json").write_text(json.dumps(chosen2))
print("validation2",chosen2)
print("levels",dict(Counter(scene_by_id[i]["level"] for i in chosen2)),
      "suites",dict(Counter(scene_by_id[i]["suite"] for i in chosen2)))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass29: visual aperture gain2.0\n"
            "Development126:75 vs70 (+4.0+/-3.3pp),11 wins/6 losses; loops44 vs46. "
            "L2 +8/86, L3 unchanged, L1 -1/4, L4 -2/18; one pre-fail proxy loss. "
            "Best development gain so far but not significant and safety unproven. "
            "Frozen for fresh validation_batch2.json; no further tuning yet.\n")
```
