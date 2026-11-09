The largest normalization outlier appears in both experts' final self-attention layer, so it may be an inherited attention feature rather than a language defect. I will instead map head-level text-attention gains and the two text encoders' input sensitivities, retaining these only as hypotheses to test against activations.
```python
t0 = time.time()
head_stats = []
for expert in ("action", "video"):
    for layer in range(30):
        p = f"mot.mixtures.{expert}.blocks.{layer}.cross_attn."
        row = {"expert": expert, "layer": layer}
        for part in ("q", "k", "v"):
            x = load_original(p + part + ".weight").reshape(24, 128, -1)
            row[part] = x.square().sum((1,2)).sqrt().numpy()
            b = load_original(p + part + ".bias").reshape(24,128)
            row[part+"_bias"] = b.norm(dim=1).numpy()
        x = load_original(p + "o.weight").reshape(-1,24,128)
        row["o"] = x.square().sum((0,2)).sqrt().numpy()
        for part in ("q", "k"):
            x = load_original(p + f"norm_{part}.weight").reshape(24,128)
            row["norm_"+part] = x.square().mean(1).sqrt().numpy()
        head_stats.append(row)
scores = []
for row in head_stats:
    # Scale-normalized, within-layer screening statistic, NOT a causal score.
    g = row["v"] * row["o"]
    qk = row["norm_q"] * row["norm_k"]
    for h in range(24):
        scores.append((float(g[h]/np.median(g)), row["expert"], row["layer"], h,
                       float(g[h]), float(qk[h]), float(row["k_bias"][h]),
                       float(row["q_bias"][h])))
print("Largest within-layer cross-attention V/O head gains:")
for x in sorted(scores, reverse=True)[:20]:
    print("relative=%.3f expert=%s layer=%d head=%d VO=%.2f normQK=%.3f |bk|=%.3f |bq|=%.3f" % x)
text_stats = {}
print("\nText embedding input-column statistics:")
for expert in ("action", "video"):
    p = f"mot.mixtures.{expert}.text_embedding."
    x = load_original(p+"0.weight")
    cn = x.norm(dim=0)
    vals, idx = cn.topk(12)
    text_stats[expert] = cn.numpy()
    print(expert, "column norm median", round(cn.median().item(),4),
          "top (channel,norm)", [(int(i), round(float(v),4)) for i,v in zip(idx,vals)])
    print(" first-layer bias rms", load_original(p+"0.bias").square().mean().sqrt().item(),
          "second-layer bias rms", load_original(p+"2.bias").square().mean().sqrt().item())
print("Column-norm correlation between experts:",
      float(np.corrcoef(text_stats["action"], text_stats["video"])[0,1]))
np.savez(W/"weight_screening.npz",
         action_text_col_norm=text_stats["action"], video_text_col_norm=text_stats["video"],
         head_scores=np.array([[s[0], int(s[1]=="video"), *s[2:]] for s in scores]))
print("Elapsed seconds:", round(time.time()-t0,1))
```
