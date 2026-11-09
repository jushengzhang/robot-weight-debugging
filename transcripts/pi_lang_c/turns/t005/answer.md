The baseline failures cluster around missing canonical object words: “cooker,” “cupboard,” “fermented grape beverage,” and descriptive phrases for bowls and drawers. I will locate the saved probe inputs and select literal controls before deciding whether the failure originates in lexical representations or in the action expert’s use of language.
```python
print("WORK top-level:", [p.name for p in W.iterdir()])
froot = W / "roll_1_frames"
print("Frame root exists:", froot.exists())
if froot.exists():
    print("Frame tree examples:", [str(p.relative_to(W)) for p in list(froot.rglob("*"))[:16]])
print("Referenced frame dir files:", [p.name for p in pathlib.Path(r1[0]["frames_dir"]).iterdir()][:12])
def task_of(v):
    n = v["name"]
    return n.split("_language_")[0] if "_language_" in n else re.split(r"_view_|_initstate_|_table_|_light_|_noise_|_layout_", n)[0]
goal_tasks = sorted(set(task_of(v) for v in lang if v["suite"] == "libero_goal"))
controls = []
for task in goal_tasks:
    pool = [v for v in variants if v["suite"] == "libero_goal" and v["category"] != "Language Instructions" and v["difficulty"] == 1 and v["name"].startswith(task+"_")]
    best = sorted(pool, key=lambda v: ({"Light Conditions":0,"Sensor Noise":1,"Background Textures":2}.get(v["category"],3),v["id"]))[0]
    controls.append(dict(suite=best["suite"],id=best["id"],seed=7))
    print(task, "CONTROL", best["id"], best["category"])
print("Controls JSON:", json.dumps(controls))
print("Successful stove paraphrases:")
for r in r1:
    if task_of(r) == "turn_on_the_stove" and r["success"]:
        print(r["id"], r["instruction"])
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass 1: diagnostic baseline\n80 hard goal paraphrases, seed 7: 49/80 (61.25%); "
            "difficulty 3 18/19, 4 11/17, 5 20/44. Failures often replace canonical object names "
            "with synonyms/descriptions. Need separate lexical encoding from expert routing.\n")
```
