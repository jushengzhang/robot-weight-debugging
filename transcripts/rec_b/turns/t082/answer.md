The earlier probes mostly tested action-expert attention. I will now check what spatial information is available upstream: the saved images, VLM activations, and vision-token structure. This should help distinguish poor visual localization from a policy that sees the scene but chooses the wrong action.
```python
def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

base_rows = read_jsonl(W / "roll_1.jsonl")
print("Baseline records:", len(base_rows))
print("Example record:", {k:v for k,v in base_rows[0].items() if k not in ("obj_deltas",)})

for d in ["probe_2", "probe_19", "probe_27", "probe_32", "roll_1_acts", "fwd"]:
    p = W / d
    print("\n", d, [q.name for q in list(p.iterdir())[:12]] if p.exists() else "missing")
    idx = p / "index.json"
    if idx.exists():
        obj = json.loads(idx.read_text())
        print("Index type:", type(obj).__name__)
        if isinstance(obj, list):
            print("First entry:", str(obj[0])[:3200])
        else:
            print("Keys:", list(obj)[:20])
            print("Excerpt:", str(obj)[:3200])

print("\nPartitions:", str(json.loads((W / "partitions.json").read_text()))[:700])
print("Relevant vision names:")
for n in NAMES:
    if any(s in n for s in ["position_embedding", "multi_modal_projector", "vision_model.post_layernorm"]):
        print(n, SHAPES[n])

sample_input = next((W / "roll_33_frames").rglob("*_inputs.pt"))
inp = torch.load(sample_input, map_location="cpu", weights_only=False)
print("Saved input:", sample_input)
print("Input structure:", {k:(tuple(v.shape),str(v.dtype)) if isinstance(v,torch.Tensor)
                           else str(type(v)) for k,v in inp.items()})
```
