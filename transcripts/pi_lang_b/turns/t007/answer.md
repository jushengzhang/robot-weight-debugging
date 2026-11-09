The saved-input filenames differ from the advertised pattern. I will inspect only the generated work directories to locate the replayable observations before requesting probes.
```python
print("roll frames entries", [(p.name, p.is_dir(), p.is_symlink()) for p in (W/"roll_1_frames").iterdir()][:8])
print("record frames", r1[0]["frames_dir"])
p=pathlib.Path(r1[0]["frames_dir"])
print("example dir entries", [(x.name,x.is_dir(),x.is_symlink()) for x in p.iterdir()][:12])
print("work roll1 generated dirs", [(x.name,x.is_dir()) for x in (W/"fwd"/"roll_1").iterdir()])
tasks_goal = sorted({v["name"].split("_language_")[0] for v in langvars if v["suite"]=="libero_goal"})
easy_goal = []
for task in tasks_goal:
    pool = sorted([v for v in langvars if v["suite"]=="libero_goal" and v["name"].startswith(task+"_language_") and v["difficulty"]<=2], key=lambda v:(v["difficulty"],v["id"]))
    easy_goal.extend(pool[:2])
print("controls", [(v["id"],v["name"].split("_language_")[0]) for v in easy_goal])
act_ex = np.load(r1[0]["activations"])
print("activation keys", [(k,act_ex[k].shape) for k in list(act_ex)[:4]])
with (W/"notes.md").open("a") as f:
    f.write("\nPass 1: original goal language d4-5, 31/64 = 48.4%; d4 13/17, d5 18/47. "
            "Descriptive object substitutions dominate failures; long context with canonical names often works.\n")
```
