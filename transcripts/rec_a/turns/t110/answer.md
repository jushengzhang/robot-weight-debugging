The layer-7 self-value edit barely reached the downstream gripper representation and changed no signs, so I will remove it. The best screened intervention—softening layer 16 head 3’s queries—remains statistically unresolved rather than successful. I will give that fixed candidate one larger, independent test on previously unused practice scenes; it must demonstrate a real gain before it can be retained.
```python
for n in list(edited()): drop_edit(n)
qw=load_original(qname); qw[768:1024]*=.5; save_edit(qname,qw)
(W/"current_candidate.json").write_text(json.dumps({"name":"E16H3_Q_0.5","layer":16,"head":3,"query_gain":.5,"status":"independent_test"}))
practice=json.loads((pathlib.Path(BENCH)/"practice_scenes.json").read_text())
print("Practice container",type(practice).__name__)
if isinstance(practice,dict):
    print("Keys",list(practice)[:10])
    scenes=practice.get("scenes",practice.get("practice_scenes"))
else: scenes=practice
print("First scene",scenes[0])
split=json.loads((W/"scene_split.json").read_text())
valid_ids=set(split["validation"])
strata=defaultdict(list)
for s in scenes:
    if s["id"] in valid_ids:
        strata[(s["level"],s["suite"])].append(s["id"])
print("Validation strata counts",dict((str(k),len(v)) for k,v in sorted(strata.items())))
import hashlib
panels=[[] for _ in range(4)]
for key,ids in sorted(strata.items()):
    ids=sorted(ids,key=lambda z:hashlib.sha256(f"independent_v1:{z}".encode()).hexdigest())
    for j in range(4):
        panels[j].extend(ids[j*4:(j+1)*4])
episodes=[[{"id":sid,"grip":gr,"seed":29} for sid in panel for gr in ["open","close"]] for panel in panels]
(W/"independent_panels.json").write_text(json.dumps(episodes,indent=2))
print("Panel sizes",[len(x) for x in episodes])
print("Panel1 episodes:",json.dumps(episodes[0]))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 34: isolated L7 action-value feedback diagnostic\n"
            "- Alpha=0.5 rank-one attenuation: 0/160 gripper signs changed, XYZ RMS 0.0020.\n"
            "- First-call grip-axis perturbation RMS L7 0.042 -> L17 0.081, small relative to natural signal. Reverted.\n"
            "- Fixed E16H3 query gain 0.5 selected for an independent test despite inconclusive discovery, not claimed beneficial.\n"
            "- Independent panels frozen by scene hash from the pre-existing validation split, balanced across level/suite, seed 29.\n")
```
