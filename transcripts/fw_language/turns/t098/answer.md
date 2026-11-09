The screening found a specific lead: action layer 11, cross-attention head 5 aligns with the instruction-divergence increment about 12.6 times more strongly than an isotropic direction. I will check whether a single output-value mode explains this alignment before constructing a narrowly targeted edit.
```python
def effective_ov_modes(expert, layer, head):
    base = f"mot.mixtures.{expert}.blocks.{layer}.cross_attn."
    o = load_original(base + "o.weight")
    v = load_original(base + "v.weight")
    hs = slice(head*128, (head+1)*128)
    oh, vh = o[:, hs], v[hs, :]
    qo, ro = torch.linalg.qr(oh, mode="reduced")
    qv, rv = torch.linalg.qr(vh.T, mode="reduced")
    us, s, vhs = torch.linalg.svd(ro @ rv.T, full_matrices=False)
    left = qo @ us
    return {"name": base+"o.weight", "o": o, "oh": oh,
            "left": left, "s": s, "right": qv @ vhs.T, "head": head}

text_target = effective_ov_modes("action", 11, 5)
d11 = contrasts[11]-contrasts[10]
d11n = d11 / torch.linalg.vector_norm(d11, dim=1, keepdim=True)
u_text = text_target["left"][:, 0].contiguous()
# Choose a sign for readable loading summaries; the projector is sign-invariant.
if u_text[torch.argmax(u_text.abs())] < 0:
    u_text = -u_text
r_text = u_text @ text_target["oh"]
sv = text_target["s"]
mode_stats = {
    "top_singular_values": [float(x) for x in sv[:5]],
    "top_OV_energy_fraction": float(sv[0]**2 / (sv**2).sum()),
    "contrast_fraction_top1_equal_pair_weight": float(((d11n @ u_text)**2).mean()),
    "contrast_fraction_top1_energy_weight": float(((d11 @ u_text)**2).sum()/(d11*d11).sum()),
    "contrast_fraction_top5_equal_pair_weight":
        float(((d11n @ text_target["left"][:, :5])**2).sum(dim=1).mean()),
    "top_output_coordinates": [
        [int(i), float(u_text[i])] for i in torch.topk(u_text.abs(), 6).indices],
    "pair_increment_mode_coefficients": [round(float(x), 5) for x in d11 @ u_text]
}
print("L11 cross-attention H5 mode evidence:", mode_stats)
torch.save({"name": text_target["name"], "head": 5, "u": u_text, "r": r_text,
            "evidence": mode_stats}, work / "action_L11H5_cross_mode_template.pt")
(work / "action_L11H5_cross_mode_evidence.json").write_text(json.dumps(mode_stats, indent=2))
print("Exact fixed-input probe paths:")
print(json.dumps([str(work_path(r["input"])) for r in p4_results], separators=(",", ":")))
```
