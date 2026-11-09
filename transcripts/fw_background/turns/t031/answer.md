The backend is online. The pilot has two background failures and no lighting-control failures, but it is too small to select an edit. I will inspect the records and activation format, then expand to the full diagnostic set.

```python
print("Top-level result files:",sorted(p.name for p in W.iterdir() if p.name.startswith(("roll_","probe_")))[:20])
roll1=[json.loads(x) for x in (W/"roll_1.jsonl").read_text().splitlines() if x.strip()]
print("Record count:",len(roll1))
for r in roll1[:3]:
    print(r)
idx1=json.loads((W/"roll_1_acts"/"index.json").read_text())
print("Activation index type:",type(idx1).__name__)
if isinstance(idx1,dict):
    print("Index keys:",list(idx1)[:12])
    print("First index value:",str(next(iter(idx1.values())))[:1600])
else:
    print("First index entries:",idx1[:2])
frame_root=W/"roll_1_frames"
print("Frame subdirectories:",[p.name for p in sorted(frame_root.iterdir())[:5]])
one_dir=next(p for p in sorted(frame_root.iterdir()) if p.is_dir())
print("Files in first frame directory:",[p.name for p in sorted(one_dir.iterdir())[:12]])
note("## Pass 1: original pilot\n"
     "- Backend now online. 12 episodes: BG 6/8 (75.0%); light 4/4 (100%).\n"
     "- Two max-step failures, both BG; sample too small for edit selection.\n"
     "- All-layer residual summaries, frames, and policy inputs saved in roll_1 artifacts.\n"
     "- Forward passes used: 1/40; weights unchanged.")
```
