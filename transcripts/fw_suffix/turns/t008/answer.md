The previous inspection stopped before defining its formatting helper. I will rerun that metadata-only inspection with the helper defined first; no weights have been changed.
```python
def describe_obj(x):
    if torch.is_tensor(x):
        return {"dtype": str(x.dtype), "shape": list(x.shape),
                "mean": round(x.float().mean().item(), 5),
                "std": round(x.float().std().item(), 5) if x.numel()>1 else 0}
    if isinstance(x, np.ndarray):
        return {"dtype": str(x.dtype), "shape": list(x.shape)}
    if isinstance(x, dict):
        return {str(k): describe_obj(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if len(x) <= 8:
            return [describe_obj(v) for v in x]
        return {"type": type(x).__name__, "length": len(x),
                "first": [describe_obj(v) for v in x[:3]]}
    return str(x)[:400]

print("Saved inputs:", len(raw_input_files), len(clean_input_files))
print("Initial input structure:", json.dumps(describe_obj(first_input), indent=2)[:4500])
print("Paired initial paths for Camera/Noise:")
for k in common:
    if rmap[k]["category"] in ["Camera Viewpoints", "Sensor Noise"] and k[0] in ["libero_goal","libero_spatial"]:
        print(str(Path(rmap[k]["frames_dir"])/"t0000_inputs.pt"))
        print(str(Path(cmap[k]["frames_dir"])/"t0000_inputs.pt"))

weight_stats = []
for expert in ["action","video"]:
    for layer in range(30):
        pre = f"mot.mixtures.{expert}.blocks.{layer}.cross_attn."
        qw = load_original(pre+"norm_q.weight")
        kw = load_original(pre+"norm_k.weight")
        ow = load_original(pre+"o.weight")
        hnorm = ow.reshape(ow.shape[0],24,128).square().sum((0,2)).sqrt()
        weight_stats.append({"expert":expert, "layer":layer,
            "q_rms":qw.square().mean().sqrt().item(),
            "k_rms":kw.square().mean().sqrt().item(),
            "q_max":qw.abs().max().item(), "k_max":kw.abs().max().item(),
            "o_norm":ow.norm().item(), "head_max_mean":(hnorm.max()/hnorm.mean()).item(),
            "max_head":hnorm.argmax().item()})
print("Cross-attention weight statistics (every third layer):")
for row in weight_stats:
    if row["layer"] % 3 == 0:
        print(row["expert"],row["layer"],{k:round(v,3) if isinstance(v,float) else v
              for k,v in row.items() if k not in ["expert","layer"]})
(work / "cross_weight_stats.json").write_text(json.dumps(weight_stats, indent=2))
```
