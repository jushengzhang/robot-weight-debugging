Full removal did not strengthen the benefit: it gained 0.8 points with a 3.1-point standard error and lost a pre-failure proxy episode. I will remove it. Next I will examine the feed-forward outputs that may repeatedly reconstruct a grasp action, rather than assuming attention concentration alone causes the loop.
```python
for n in list(edited()):
    drop_edit(n)
for k in [2,8,12,15,16]:
    obj=json.loads((W/f"probe_{k}/results.json").read_text())
    print("probe",k,"type",type(obj).__name__,"sample",str(obj)[:1600] if k in [8,16] else str(obj)[:300])
r=json.loads((W/"fwd/roll_1/results.jsonl").read_text().splitlines()[0])
print("raw rollout keys",list(r))
print("raw record",str(r)[:1500])
for sid,grip in [(979,"close"),(979,"open"),(562,"close"),(562,"open"),(639,"open")]:
    p=W/f"roll_1_frames/s{sid}_{grip}_seed7_original"
    print(p.name, sorted(x.name for x in p.iterdir())[:32] if p.exists() else "missing")
z=np.load(next((W/"roll_1_acts").glob("*.npz")))
print("rollout activation keys",[k for k in z.files if k.startswith("mean")])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass18: full visual aperture removal\n"
            "71 vs70/126 (+0.8+/-3.1pp),8 wins/7 losses; loops44 vs46, mean flips15.27 vs14.71. "
            "No useful dose response, pre-fail loss. Removed all aperture edits; not a demonstrated repair.\n")
```
