I will inspect the failed instructions and the saved-input format. I will also choose easier paraphrases of the same ten tasks as controls, rather than comparing unrelated scenes.
```python
def read_roll(k):
    return [json.loads(x) for x in (W / f"roll_{k}.jsonl").read_text().splitlines() if x.strip()]
r1 = read_roll(1)
print("record keys", list(r1[0]))
print("first record", json.dumps(r1[0], ensure_ascii=True)[:3800])
print("activation index", (W/"roll_1_acts"/"index.json").read_text()[:1900])
print("\nInstructions")
for r in r1:
    print(json.dumps({k:r[k] for k in r if k in ["id","variant_id","seed","success","instruction","failure_mode","task","suite"]}, ensure_ascii=True))
frame_dirs = sorted((W/"roll_1_frames").iterdir())
print("frame dirs", [p.name for p in frame_dirs[:3]])
print("files", [p.name for p in frame_dirs[0].iterdir()][:8])
probe_example = sorted((W/"roll_1_frames").rglob("*_inputs.pt"))[0]
inp = torch.load(probe_example, map_location="cpu", weights_only=False)
def structure(x):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, np.ndarray):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {k: structure(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [structure(v) for v in x[:4]]
    return str(x)[:300]
print("input", str(probe_example), structure(inp))
tasks_goal = sorted({v["name"].split("_language_")[0] for v in langvars if v["suite"]=="libero_goal"})
easy_goal = []
for task in tasks_goal:
    pool = sorted([v for v in langvars if v["suite"]=="libero_goal" and v["name"].startswith(task+"_language_") and v["difficulty"]<=2], key=lambda v:(v["difficulty"],v["id"]))
    print("control candidates", task, [(v["id"],v["difficulty"]) for v in pool[:4]])
    easy_goal.extend(pool[:2])
```
