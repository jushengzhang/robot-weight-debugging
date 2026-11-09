The first run reproduces both reported problems: 9 of 12 task failures reached the old goal, and 22 of 36 failures had re-grasp loops. I will inspect the matched records, action traces, and saved-input structure before choosing attention layers to probe.
```python
def read_jsonl(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]
r1 = read_jsonl(W / "roll_1.jsonl")
print("record fields:", list(r1[0]))
print("example:", r1[0])
idx1 = json.loads((W / "roll_1_acts/index.json").read_text())
print("activation index type:", type(idx1).__name__)
print("activation index preview:", str(idx1)[:3000])
print("\nmatched outcomes:")
for j in range(0,len(r1),5):
    group = r1[j:j+5]
    print([(r.get("id",r.get("episode_id")),r.get("ptype"),r["success"],r.get("orig_goal_reached"),r.get("grip_flips"),r.get("steps")) for r in group])
paths1 = sorted((W/"roll_1_frames").rglob("*inputs.pt"))
print("n stored inputs:", len(paths1))
print("first paths:", [str(p.relative_to(W)) for p in paths1[:8]])
inp1 = torch.load(paths1[0], map_location="cpu", weights_only=False)
def structure(x, depth=0):
    if isinstance(x,dict):
        return {k:structure(v,depth+1) for k,v in x.items()}
    if isinstance(x,(torch.Tensor,np.ndarray)):
        return (type(x).__name__,tuple(x.shape),str(x.dtype))
    if isinstance(x,(list,tuple)) and len(x)>12:
        return (type(x).__name__,len(x),str(x[:3])[:100])
    return str(x)[:200]
print("input structure:", structure(inp1))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 1: matched diagnostic baseline\n80 episodes / 16 scenes, seed17: 55% success. Task 4/16, position 6/16, environment 6/16, object 13/16, semantic 15/16. 9/12 task failures reached original goal. 22/36 failures had >=6 grip flips. No edits yet.\n")
```
