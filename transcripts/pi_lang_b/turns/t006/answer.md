I will finish the compact metadata inspection and select two low-difficulty language controls per goal task. I will keep output short enough that the diagnostic execution completes.
```python
frame_dirs = sorted((W/"roll_1_frames").iterdir())
probe_example = sorted((W/"roll_1_frames").rglob("*_inputs.pt"))[0]
inp = torch.load(probe_example, map_location="cpu", weights_only=False)
def structure(x):
    if isinstance(x, (torch.Tensor,np.ndarray)):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {k: structure(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [structure(v) for v in x[:3]]
    return str(x)[:120]
print("input file", str(probe_example).replace(str(W),"WORK"))
print("input structure", json.dumps(structure(inp), ensure_ascii=True)[:2000])
tasks_goal = sorted({v["name"].split("_language_")[0] for v in langvars if v["suite"]=="libero_goal"})
easy_goal = []
for task in tasks_goal:
    pool = sorted([v for v in langvars if v["suite"]=="libero_goal" and v["name"].startswith(task+"_language_") and v["difficulty"]<=2], key=lambda v:(v["difficulty"],v["id"]))
    easy_goal.extend(pool[:2])
print("controls", [(v["id"],v["name"].split("_language_")[0]) for v in easy_goal])
act_ex = np.load(r1[0]["activations"])
print("activation keys", [(k,act_ex[k].shape) for k in list(act_ex)[:5]])
with (W/"notes.md").open("a") as f:
    f.write("\nPass 1: original goal language d4-5, 31/64 = 48.4%; d4 13/17, d5 18/47. "
            "Descriptive object substitutions dominate failures; long context with canonical names often works.\n")
```
