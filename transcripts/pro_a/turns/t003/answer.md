The baseline reproduces the main failure: only 5/27 task perturbations succeed, and 13 failures reach the old scene goal. Position succeeds on only 3/26. I will now inspect the saved records and replay-input format, select matched diagnostic cases, and reserve fresh episodes for validation.
```python
import hashlib
def read_jsonl(path):
    return [json.loads(s) for s in pathlib.Path(path).read_text().splitlines() if s.strip()]
r1 = read_jsonl(W / "roll_1.jsonl")
print("Record example:", r1[0])
ai1 = json.loads((W / "roll_1_acts/index.json").read_text())
print("Activation index type:", type(ai1).__name__)
print("Activation index excerpt:", str(ai1)[:2600])
frame_dirs = sorted((W / "roll_1_frames").iterdir())
print("Frame directory example:", frame_dirs[0].name)
print("Contents:", [p.name for p in sorted(frame_dirs[0].iterdir())][:12])
input_path = sorted(frame_dirs[0].glob("*inputs.pt"))[0]
inp = torch.load(input_path, map_location="cpu", weights_only=False)
def describe(x, depth=0):
    if isinstance(x, dict):
        return {str(k): describe(v,depth+1) for k,v in x.items()}
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return {"shape":list(x.shape), "dtype":str(x.dtype)}
    if isinstance(x, (list,tuple)):
        return [describe(v,depth+1) for v in x[:4]] + (["..."] if len(x)>4 else [])
    return str(x)[:180]
print("Saved inputs:", describe(inp))
used1 = {int(r.get("id",r.get("episode_id",-1))) for r in r1}
print("Record fields:", list(r1[0]))
print("Old-task failures:")
for r in r1:
    if r.get("ptype")=="task" and not r.get("success") and r.get("orig_goal_reached"):
        print({k:r.get(k) for k in ["id","episode_id","suite","task","instruction","steps","grip_flips","eef_path_m","obj_moved_m"]})
def partition(e):
    key = f'{e["suite"]}/{e["task"]}/{e["init_idx"]}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16) % 5
byid = {e["id"]:e for e in episodes}
dev_pool = [e for e in episodes if partition(e) in (0,1,2)]
val_pool = [e for e in episodes if partition(e)==3 and e["id"] not in used1]
final_pool = [e for e in episodes if partition(e)==4 and e["id"] not in used1]
print("Development / validation / final pool:", len(dev_pool),len(val_pool),len(final_pool))
(W/"partitions.json").write_text(json.dumps({
    "development":[e["id"] for e in dev_pool],
    "validation":[e["id"] for e in val_pool],
    "final":[e["id"] for e in final_pool],
    "baseline_ids":sorted(used1)}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 1: baseline (128 episodes, sampler seed 1701)\n"
            "Environment 17/25, object 23/26, position 3/26, semantic 21/24, task 5/27; "
            "13 task failures reach original goal; 28/59 failures have regrasp loops. "
            "Sampler ignored 'original' (not in provided practice file). Original-task testing currently unavailable.\n"
            "Partition by SHA256(suite/task/init_idx) mod 5: 0-2 development; 3 validation; 4 final, "
            "excluding baseline episodes from validation/final.\n")
```
