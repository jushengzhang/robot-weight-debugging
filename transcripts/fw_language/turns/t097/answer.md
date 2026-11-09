The query biases are small, so they do not provide a convincing target. I will instead check which direct text-attention head outputs align with the measured mid-layer instruction differences. This is only a screening step; any resulting edit still needs a behavioral test.
```python
torch.set_num_threads(8)
p4_acts = []
for r in p4_results:
    with np.load(work_path(r["activations"])) as z:
        p4_acts.append({l: torch.from_numpy(z[f"mean::action.{l}.resid::action"].copy()).float()
                       for l in range(30)})
assert len(p4_acts) == 20
contrasts = {l: torch.stack([p4_acts[2*j+1][l]-p4_acts[2*j][l]
                             for j in range(10)]) for l in range(30)}
temp_lookup = {(x["layer"], x["head"]): x for x in action_screen}
alignment_screen = []
for l in [10, 11, 12, 13]:
    d = contrasts[l] - contrasts[l-1]
    dn = d / torch.linalg.vector_norm(d, dim=1, keepdim=True).clamp_min(1e-12)
    o = load_original(f"mot.mixtures.action.blocks.{l}.cross_attn.o.weight")
    for h in range(24):
        oh = o[:, h*128:(h+1)*128]
        energy_alignment = float(1024 * ((dn @ oh)**2).sum(dim=1).mean() /
                                 (oh**2).sum())
        basis = torch.linalg.qr(oh, mode="reduced").Q
        span_fraction = float(((dn @ basis)**2).sum(dim=1).mean())
        t = temp_lookup[(l, h)]
        alignment_screen.append({
            "layer": l, "head": h,
            "energy_alignment_vs_isotropic": energy_alignment,
            "span_fraction": span_fraction,
            "temperature_proxy": t["temperature_proxy"],
            "temperature_relative": t["relative"]
        })
    print("Layer", l, "increment RMS", float(torch.sqrt((d*d).mean())),
          "top output alignments:",
          sorted([r for r in alignment_screen if r["layer"] == l],
                 key=lambda r: r["energy_alignment_vs_isotropic"], reverse=True)[:4])
(work / "direct_text_head_alignment_discovery.json").write_text(
    json.dumps(alignment_screen, indent=2))
```
