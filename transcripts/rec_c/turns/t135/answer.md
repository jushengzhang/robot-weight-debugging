The head-1-only probe is more selective: reserved empty closures gain 0.137 vertical action units, while contact and open means stay within 0.017 and gripper signs are unchanged. I will now freeze the candidate and audit the remaining scene pool, keeping untouched confirmation scenes separate from the pre-fail proxies.
```python
practice = json.loads((Path(BENCH)/"practice_scenes.json").read_text())
if isinstance(practice, dict):
    print("practice top-level keys", list(practice))
    scenes = practice.get("scenes", practice.get("practice_scenes"))
else:
    scenes = practice
print("practice count",len(scenes))
print("sample scene",scenes[0])
print("stages",Counter(r.get("stage","?") for r in scenes))
r0 = json.loads((W/"roll_40.jsonl").read_text().splitlines()[0])
print("roll keys",list(r0))
print("roll identity", {k:v for k,v in r0.items() if k in ["id","scene_id","grip","seed","model","arm","success"]})
print("seen ID count",len(seen), "seen ID types",Counter(type(x).__name__ for x in seen))
scene_by_id = {r["id"]:r for r in scenes}
print("confirmation strata",Counter((scene_by_id[i]["level"],scene_by_id[i]["stage"]) for i in confirm3))
req45 = {"type":"rollout","model":"both",
         "episodes":[{"id":i,"grip":g,"seed":5006} for i in confirm3 for g in ["open","close"]],
         "save_frames":True}
(W/"frozen_request45.json").write_text(json.dumps(req45))
with (W/"notes.md").open("a") as f:
    f.write("H1-only0.60 frozen before outcomes for confirmation_batch3,64 scenes,both grips,seed5006. "
            "Fixed-query selectivity improved; this is not yet evidence of recovery success.\n")
```
