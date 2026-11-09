The rollout records are stored in completion order rather than request order, so I will match them by episode ID. I will also locate the actual replay-input files and summarize object motion for failed new-instruction cases; reaching the old goal is stronger evidence than action similarity alone.
```python
rec1 = {r["id"]:r for r in r1}
print("frame directory entries:", [str(p.relative_to(W)) for p in (W/"roll_1_frames").rglob("*")][:35])
print("fwd entries:", [str(p.relative_to(W)) for p in (W/"fwd/roll_1").iterdir()][:15])
print("\nmatched outcomes [object, position, semantic, task, environment]:")
for suite,t,i in chosen_scenes:
    rr=[rec1[by_scene[(suite,t)][i][p]["id"]] for p in ptypes]
    print(t[:64], [(x["id"],int(x["success"]),x["grip_flips"]) for x in rr])
print("\nfailed task object motion:")
for r in r1:
    if r["ptype"]=="task" and not r["success"]:
        a=np.load(r["actions_npy"])
        print(r["id"],r["instruction"],"old_goal",r["orig_goal_reached"])
        print("  moved",r["obj_moved_m"],"action shape",a.shape,"first translation",np.round(a[:5,:3].mean(0),3).tolist())
act0=np.load(r1[0]["activations"])
print("activation file keys:", act0.files[:16], "shapes", [(k,act0[k].shape) for k in act0.files[:5]])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 1: matched diagnostic baseline\n80 episodes / 16 scenes, seed17: 55% success. Task 4/16, position 6/16, environment 6/16, object 13/16, semantic 15/16. 9/12 task failures reached original goal. 22/36 failures had >=6 grip flips. No edits yet.\n")
```
