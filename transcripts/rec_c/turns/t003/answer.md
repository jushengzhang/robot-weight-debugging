The baseline has 46 looping failures out of 56 failures, so the reported pathology is present in this sample. I will inspect the artifact format, find same-scene success/failure pairs, and compare layer summaries while separating loop failures from other failures.

```python
def wp(p):
    p = str(p)
    return W / p[5:] if p.startswith("WORK/") else pathlib.Path(p)
def read_roll(k):
    with open(W / f"roll_{k}.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]
r1 = read_roll(1)
print("record keys:", list(r1[0]))
print("example record:", json.dumps(r1[0], ensure_ascii=True)[:6000])
with open(W / "roll_1_acts/index.json") as f:
    idx1 = json.load(f)
print("index type:", type(idx1).__name__)
print("index sample:", json.dumps(idx1 if not isinstance(idx1, list) else idx1[:1], ensure_ascii=True)[:3500])
print("WORK entries:", sorted(p.name for p in W.iterdir()))
print("frame dirs:", [p.name for p in (W / "roll_1_frames").iterdir()][:5])
print("failures, compact:")
for r in r1:
    if not r.get("success", False) and not r.get("error"):
        print({k:r.get(k) for k in ["id","scene_id","suite","level","grip","seed","steps","grip_flips","eef_path_m","eef_net_m"]})
        if sum(1 for a in r1[:r1.index(r)+1] if not a.get("success",False) and not a.get("error")) >= 10:
            break
with open(W / "notes.md", "a") as f:
    f.write("\n## Pass 1 baseline\n128 requested; 126 valid. Success 70/126 (55.6%); 46/56 failures have >=6 gripper flips. L2 48/86, L3 10/18, L4 9/18, L1 3/4. Saved actions, observations, and attention-output summaries. No edits.\n")
```
