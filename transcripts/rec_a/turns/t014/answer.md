I will summarize the late heads and compare their camera bias with the spatial heads identified earlier. I will then construct a fixed discovery screening panel; all candidate interventions will use exactly the same scenes, seeds, and gripper starts, with successful baseline episodes retained as damage controls.
```python
p5=json.loads((W/"probe_5/results.json").read_text())
att5={p["idx"]:{int(l):np.load(fp).astype(np.float32).mean(1) for l,fp in p["attention"].items()} for p in p5}
print("Layer/head loop B/W/lang/self -> success B/W/lang/self")
for l in [9,11,13,16]:
    AL=(att5[0][l]+att5[2][l])/2; AS=(att5[1][l]+att5[3][l])/2
    for h in range(8):
        def ms(A): return [A[h,:256].sum(),A[h,256:512].sum(),A[h,768:].sum(),1-A[h].sum()]
        print(f"{l:2d}/{h}", "/".join(f"{v:.2f}" for v in ms(AL)), "->", "/".join(f"{v:.2f}" for v in ms(AS)))
print("VLM input norm weight means/std:")
for l in [8,10,12,16]:
    v=load_original(f"{VL}.layers.{l}.input_layernorm.weight")
    print(l,round(float(v.mean()),3),round(float(v.std()),3))
# Outcome-independent subset: first eight originally selected scenes at each level.
screen_ids = [i for lv in ["L1","L2","L3","L4"] for i in [j for j in diag_ids if scene_by_id[j]["level"]==lv][:8]]
screen_episodes=[{"id":i,"grip":g,"seed":17} for i in screen_ids for g in ["open","close"]]
screen0=[r for r in r1 if r["id"] in screen_ids]
print("Screen IDs:",screen_ids)
print("Screen baseline:",len(screen0),sum(r["success"] for r in screen0),"success;",
      sum(not r["success"] and r["grip_flips"]>=6 for r in screen0),"loop failures")
(W/"screen_episodes.json").write_text(json.dumps(screen_episodes))
print("Screen request:",json.dumps({"type":"rollout","model":"both","episodes":screen_episodes}))
```
