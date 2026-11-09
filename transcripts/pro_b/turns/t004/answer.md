The baseline reproduces the expected pattern: 10 of 11 task failures reach the old goal, while position/environment failures often loop. I’ll inspect matched action traces and saved inputs, then probe attention on identical observations so that later behavioral differences do not contaminate the diagnosis.

```python
def workpath(p):
    p = str(p)
    return W / p[5:] if p.startswith("WORK/") else Path(p)

def read_roll(k):
    return [json.loads(s) for s in (W / f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]

r1 = read_roll(1)
print("record example:", r1[0])
idx1 = json.loads((W / "roll_1_acts" / "index.json").read_text())
print("activation index type:", type(idx1).__name__)
print("index example:", str(idx1)[:2200])
print("frame subdirs:", [p.name for p in (W / "roll_1_frames").iterdir()][:6])
print("\nmatched behavior:")
for g in diag_groups:
    es = groups[g]
    rows = []
    for p in ("object", "position", "semantic", "task", "environment"):
        e = es[p]
        rr = next((r for r in r1 if r.get("id",r.get("episode_id")) == e["id"]), None)
        if rr:
            rows.append((p, rr.get("success"), rr.get("orig_goal_reached"), rr.get("steps"),
                         rr.get("grip_flips"), rr.get("eef_path_m"), rr.get("obj_moved_m")))
    print(g[0], g[1][:43], rows)

input_paths = sorted((W / "roll_1_frames").glob("*/*_inputs.pt"))
print("\ninput files:", len(input_paths), [str(p.relative_to(W)) for p in input_paths[:3]])
x0 = torch.load(input_paths[0], map_location="cpu", weights_only=False)
def compact_tree(x, depth=0):
    if isinstance(x, torch.Tensor):
        return {"shape":list(x.shape), "dtype":str(x.dtype)}
    if isinstance(x, np.ndarray):
        return {"shape":list(x.shape), "dtype":str(x.dtype)}
    if isinstance(x, dict):
        return {str(k):compact_tree(v,depth+1) for k,v in x.items()}
    if isinstance(x,(list,tuple)) and len(x)>10:
        return {"type":type(x).__name__,"len":len(x),"first":compact_tree(x[0],depth+1)}
    return str(x)[:300]
print("policy input structure:", compact_tree(x0))
with (W / "notes.md").open("a") as f:
    f.write("\n## Pass 1 baseline (75 matched episodes, seed 17)\n"
            "Overall 44/75; object 14/15, semantic 15/15, task 4/15, position 4/15, environment 7/15. "
            "10/11 task failures reached old goal; 14/31 failures had >=6 gripper flips. "
            "Saved frames, inputs, residual summaries at layers 0,3,6,9,12,15,17.\n")
```
