The baseline has 57 failures, 41 of them re-grasp loops, with the largest deficits in L3 and L4. I will now inspect the saved record format, compare motion and gripper traces, and rank layer-level activation differences; I will also look for same-scene pairs where one gripper start succeeds and the other loops.
```python
def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]

r1 = read_jsonl(W / "roll_1.jsonl")
idx1 = json.loads((W / "roll_1_acts" / "index.json").read_text())
print("Record example:", json.dumps(r1[0], indent=2)[:6500])
print("Activation index type:", type(idx1).__name__)
print("Activation index sample:", str(idx1)[:2500])
print("Frame folders:", [p.name for p in sorted((W/"roll_1_frames").iterdir())[:6]])
print("Activation directory sample:", [p.name for p in sorted((W/"roll_1_acts").iterdir())[:6]])
print("Record keys:", sorted(set().union(*(r.keys() for r in r1))))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 1: original, discovery, 128 paired-start episodes\n"
            "- Success 71/128 (55.5%); L1 22/32, L2 21/32, L3 15/32, L4 13/32.\n"
            "- Failures 57; re-grasp-loop failures 41; already-solved starts 7.\n"
            "- Full expert attention-output summaries, selected VLM/vision summaries, "
            "actions and frames saved. No edits.\n")
```
