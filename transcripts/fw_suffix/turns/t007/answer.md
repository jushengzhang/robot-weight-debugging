The saved frame directories are symlinks within WORK, so I can access their replay files directly. I will inspect initial-input metadata and cross-attention weight norms to choose informative layers for head-level probes, without running any part of the network locally.
```python
raw_input_files = sorted(p for r in raw_records for p in Path(r["frames_dir"]).glob("*_inputs.pt"))
clean_input_files = sorted(p for r in clean_records for p in Path(r["frames_dir"]).glob("*_inputs.pt"))
first_raw_paths = [Path(rmap[k]["frames_dir"]) / "t0000_inputs.pt" for k in common]
first_clean_paths = [Path(cmap[k]["frames_dir"]) / "t0000_inputs.pt" for k in common]
first_input = torch.load(first_raw_paths[0], map_location="cpu", weights_only=False)
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
        # Output columns are the individual attention-head output channels.
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
