The tag caused a 16.7-point drop on these 24 episodes, although this sample is too small to judge repairs. I’ll inspect the saved input and activation formats, verify that the initial raw/clean observations match, and assemble a compact replay set for controlled tests.```python
def read_jsonl(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
r1 = read_jsonl(W / "roll_1.jsonl")
r2 = read_jsonl(W / "roll_2.jsonl")
print("Episode record example:", str(r1[0])[:2500])
print("Activation index preview:", (W / "roll_1_acts" / "index.json").read_text()[:1800])
raw_paths, clean_paths = [], []
for v in splits["diagnostic"]:
    sub = f"{v['suite']}_v{v['id']}_s{v['seed']}_original"
    raw_paths.append(str(W / "roll_1_frames" / sub / "t0000_inputs.pt"))
    clean_paths.append(str(W / "roll_2_frames" / sub / "t0000_inputs.pt"))
p0 = torch.load(raw_paths[0], weights_only=False, map_location="cpu")
p1 = torch.load(clean_paths[0], weights_only=False, map_location="cpu")
def schema(x):
    if isinstance(x, torch.Tensor):
        return ("tensor", tuple(x.shape), str(x.dtype))
    if isinstance(x, np.ndarray):
        return ("array", x.shape, str(x.dtype))
    if isinstance(x, dict):
        return {k: schema(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [schema(v) for v in x[:4]]
    return str(x)[:600]
print("Policy input schema:", schema(p0))
print("Clean schema:", schema(p1))
print("Language diagnostic episodes:", json.dumps(eps("language_diag")))
with (W / "notes.md").open("a") as f:
    f.write("\nPasses1-2: original 24 matched raw/strip episodes: 13/24 vs17/24 "
            "(54.2% vs70.8%). Frames/replay inputs and both-expert attn_out summaries saved. "
            "Camera0->1, Background2->4, Layout2->3; others unchanged in count.\n")
```
