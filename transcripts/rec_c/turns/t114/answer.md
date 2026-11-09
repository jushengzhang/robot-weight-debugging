The larger dose gives a strongly dose-dependent vertical effect (slope 3.54, correlation 0.99) with no gripper-sign changes in the probe. It also causes modest extra descent in normal controls, which is a safety concern. I will freeze this dose and test it on fresh scenes now, rather than selecting it from another run on the development set.
```python
seen=set()
for p in W.glob("roll_*.jsonl"):
    for line in p.read_text().splitlines():
        if line.strip(): seen.add(json.loads(line)["id"])
# Also exclude every source/control scene already inspected through original saved rollouts.
seen.update(m["id"] for m in obsmeta)
remaining=[s for s in scenes if s["id"] not in seen]
rng=np.random.default_rng(400600)
chosen=rng.choice(len(remaining),128,replace=False)
validation3=[remaining[int(i)]["id"] for i in chosen[:64]]
confirmation3=[remaining[int(i)]["id"] for i in chosen[64:]]
(W/"validation_batch3.json").write_text(json.dumps(validation3))
(W/"confirmation_batch3.json").write_text(json.dumps(confirmation3))
lookup={s["id"]:s for s in scenes}
print("validation3",validation3)
print("levels",dict(collections.Counter(lookup[i]["level"] for i in validation3)),
      "suites",dict(collections.Counter(lookup[i]["suite"] for i in validation3)))
print("uninspected remaining scenes",len(remaining),"reserved confirmation",len(confirmation3))
with (W/"notes.md").open("a") as f:
    f.write("\nClearance0.60 frozen for fresh validation_batch3.json (64 scenes,both grips,seed4006). "
            "Excluded all previously rolled scenes and diagnostic inputs. "
            "confirmation_batch3.json reserved separately before observing validation results.\n")
```
