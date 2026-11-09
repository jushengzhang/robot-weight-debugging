The simulator is still offline, so I will use this time to characterize the attention weights without running the model. I will look for unusually strong or bias-dominated action readers and video key/value heads, and check the patch embedding's spatial sensitivity. These are candidates to test later, not evidence of a failure mechanism by themselves.```python
t0 = time.time()
weight_head_stats = []
def nm(expert, layer, suffix):
    return f"mot.mixtures.{expert}.blocks.{layer}.{suffix}"
def head_rms(t):
    return t.reshape(24, 128, -1).square().mean((1,2)).sqrt().numpy()
for layer in range(30):
    row = {"layer": layer}
    for expert in ("action", "video"):
        p = f"{expert}_"
        for proj in ("q", "k", "v"):
            wt = load_original(nm(expert, layer, f"self_attn.{proj}.weight"))
            bi = load_original(nm(expert, layer, f"self_attn.{proj}.bias"))
            row[p+proj+"_row_norm"] = wt.reshape(24, 128, -1).square().sum(2).mean(1).sqrt().numpy()
            row[p+proj+"_bias_rms"] = head_rms(bi)
            del wt, bi
        for qk in ("q", "k"):
            g = load_original(nm(expert, layer, f"self_attn.norm_{qk}.weight"))
            row[p+qk+"_gain"] = head_rms(g)
            del g
        out = load_original(nm(expert, layer, "self_attn.o.weight"))
        row[p+"o_head_norm"] = out.reshape(out.shape[0],24,128).square().sum((0,2)).sqrt().numpy()
        del out
    weight_head_stats.append(row)
np.savez(W / "weight_head_stats.npz", **{
    k: np.stack([r[k] for r in weight_head_stats]) for k in weight_head_stats[0] if k != "layer"
})
hs = {k: np.stack([r[k] for r in weight_head_stats]) for k in weight_head_stats[0] if k != "layer"}
def print_top_heads(label, a, n=12):
    print(label, "min/median/max:", np.round([a.min(), np.median(a), a.max()],4).tolist())
    for idx in np.argsort(a.ravel())[-n:][::-1]:
        l,h = np.unravel_index(idx, a.shape)
        print("  L%02d H%02d %.5g" % (l,h,a[l,h]))
print("Elapsed seconds:", round(time.time()-t0,1))
print_top_heads("Action q bias / row norm", hs["action_q_bias_rms"] / (hs["action_q_row_norm"]+1e-8))
print_top_heads("Action q RMS gain", hs["action_q_gain"])
print_top_heads("Video k RMS gain", hs["video_k_gain"])
reader_proxy = hs["action_o_head_norm"] * hs["video_v_row_norm"] * hs["action_q_gain"] * hs["video_k_gain"]
print_top_heads("Weight-only reader strength proxy", reader_proxy)

patch = load_original("mot.mixtures.video.patch_embedding.weight").reshape(3072,48,2,2)
spatial_mean = patch.mean((2,3), keepdim=True)
print("Patch spatial-contrast energy fraction:",
      round(float((patch-spatial_mean).square().sum()/patch.square().sum()),4))
psv = torch.linalg.svdvals(patch.flatten(1))
print("Patch singular values, first 12:", [round(x,3) for x in psv[:12].tolist()])
print("Patch energy in top 16/32/64:",
      [round(float(psv[:k].square().sum()/psv.square().sum()),4) for k in (16,32,64)])
del patch, spatial_mean, psv
print("\nObject task/category naming examples:")
for cat in ("Camera Viewpoints", "Light Conditions"):
    for x in [x for x in discovery if x["suite"]=="libero_object" and x["category"]==cat][:5]:
        print(x["id"], x["difficulty"], task_name(x), "FULL:", x["name"])
with open(W / "notes.md", "a") as f:
    f.write("\n## Setup\n- Forward backend initially offline; no passes spent.\n"
            "- Practice variants partitioned by SHA256 into 40% discovery and 60% untouched validation.\n"
            "- Initial matched panel: 20 camera and 10 mild-light episodes across 10 tasks.\n"
            "- Weight-only self-attention head statistics saved; these are hypotheses, not causal findings.\n")
```
