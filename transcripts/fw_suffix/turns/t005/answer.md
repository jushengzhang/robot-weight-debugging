The diagnostic success rate rises from 13/24 with tags to 20/24 without them. I will now inspect the saved records and replay-input format, and compare layerwise attention-output changes; episode-averaged differences are only a screening signal because the trajectories have diverged.
```python
def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]

raw_records = read_jsonl(work / "roll_1.jsonl")
clean_records = read_jsonl(work / "roll_2.jsonl")
print("Raw record example:", json.dumps(raw_records[0])[:2400])
print("Clean record example:", json.dumps(clean_records[0])[:1200])
act_idx = {}
for run in [1, 2]:
    idx = json.loads((work / f"roll_{run}_acts/index.json").read_text())
    act_idx[run] = idx
    print(f"Activation index {run}: type={type(idx).__name__}")
    print(str(idx)[:2400])

raw_input_files = sorted((work / "roll_1_frames").rglob("*_inputs.pt"))
clean_input_files = sorted((work / "roll_2_frames").rglob("*_inputs.pt"))
print("Saved input counts:", len(raw_input_files), len(clean_input_files))
print("First raw input path:", str(raw_input_files[0]))

def describe_obj(x, depth=0):
    if torch.is_tensor(x):
        return {"type": str(x.dtype), "shape": list(x.shape),
                "mean": round(x.float().mean().item(), 5),
                "std": round(x.float().std().item(), 5) if x.numel()>1 else 0}
    if isinstance(x, np.ndarray):
        return {"type": str(x.dtype), "shape": list(x.shape)}
    if isinstance(x, dict):
        return {str(k): describe_obj(v, depth+1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if len(x) <= 8:
            return [describe_obj(v, depth+1) for v in x]
        return {"type": type(x).__name__, "length": len(x),
                "first": [describe_obj(v, depth+1) for v in x[:3]]}
    return str(x)[:400]

first_input = torch.load(raw_input_files[0], map_location="cpu", weights_only=False)
print("Input structure:", json.dumps(describe_obj(first_input), indent=2)[:4000])

first_raw_paths = sorted([p for p in raw_input_files if p.name.startswith("t0000")])
first_clean_paths = sorted([p for p in clean_input_files if p.name.startswith("t0000")])
print("Initial observation pairs:", len(first_raw_paths), len(first_clean_paths))
print("Initial filenames:", [p.name for p in raw_input_files[:4]])
with (work / "notes.md").open("a") as f:
    f.write("\n### Passes 1-2: original diagnostic raw vs strip\n"
            "24 matched episodes, raw 13/24 (54.17%), strip 20/24 (83.33%). "
            "Raw/strip by category: Background 2/4->3/4; Camera 0/4->2/4; "
            "Light 4/4->4/4; Layout 2/4->3/4; Initial States 3/4->4/4; "
            "Noise 2/4->4/4. All-layer attn_out and replay inputs saved.\n")
```
