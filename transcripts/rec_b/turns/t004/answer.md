The baseline reproduces the reported pattern: 35 of 39 failures are re-grasp loops, with L3 particularly weak. I will now compare those loops with successful episodes, inspect action repetition and object displacement, and locate the activation files needed for more focused probes.
```python
def read_roll(k):
    return [json.loads(line) for line in (W/f"roll_{k}.jsonl").read_text().splitlines() if line.strip()]
r1=read_roll(1)
print("record example:",json.dumps(r1[0],ensure_ascii=True)[:5000])
ix1=json.loads((W/"roll_1_acts/index.json").read_text())
print("activation index type",type(ix1).__name__)
print("activation index example:",str(ix1)[:3500])
print("frame directories:",[p.name for p in (W/"roll_1_frames").iterdir()][:6])
print("episode summary:")
for r in r1:
    if not r.get("success") or r.get("id") in [322,647,815,1340]:
        print({k:r.get(k) for k in ["id","scene_id","grip","success","steps","grip_flips","frac_grip_closed","eef_path_m","eef_net_m","obj_deltas"]})
print("WORK files:",[p.name for p in W.iterdir()][:25])
```
