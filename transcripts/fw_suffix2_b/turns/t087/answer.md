Rank 320 did not improve the development result, so I won’t increase the projection rank further. One remaining question is whether the old single-feature gate misses suffix-like context samples. I’ll check that using the saved readouts, while distinguishing clearly task-like samples from unmatched ones.
```python
Q = P["Qprotect"].double()
R, C, L = X[:24], X[24:48], X[48:]
sim = torch.bmm(
    R / R.norm(dim=-1, keepdim=True).clamp_min(1e-12),
    (C / C.norm(dim=-1, keepdim=True).clamp_min(1e-12)).transpose(1,2)
)
bestcos = sim.max(-1).values
old_u = art["readout_instrument_meta.pt"]["u"].double()
old_scores = R @ old_u
cats = [r["category"] for r in sp["diagnostic"]]
catlist = sorted(set(cats))
new_dirs = []
for cat in catlist:
    mask = torch.tensor([c == cat for c in cats])[:, None].expand(24,30)
    z = R[mask & (bestcos < .3)]
    if len(z) == 0:
        z = R[mask & (bestcos < .5)]
    z = z - (z @ Q) @ Q.T
    v = z.mean(0)
    v /= v.norm()
    new_dirs.append(v)
new_dirs = torch.stack(new_dirs)
NS = R @ new_dirs.T
print("category | novel count | old score median/max | new strongest median/max")
for cat in catlist:
    ix = torch.tensor([c == cat for c in cats])[:, None].expand(24,30) & (bestcos < .5)
    a, b = old_scores[ix], NS.max(-1).values[ix]
    print(cat, int(ix.sum()),
          [round(float(a.median()),3), round(float(a.max()),3)],
          [round(float(b.median()),3), round(float(b.max()),3)])
for label, mask in (("clearly task-like",bestcos > .8),
                    ("mixed", (bestcos >= .5) & (bestcos <= .8)),
                    ("unmatched",bestcos < .5)):
    a = NS.max(-1).values[mask]
    print(label, "n", len(a), "new max-score quantiles",
          [round(float(t),4) for t in torch.quantile(a, torch.tensor([0.,.5,.9,1.],dtype=torch.float64))])
print("Clean new-score range:", float((C@new_dirs.T).min()), float((C@new_dirs.T).max()))
print("Language new-score range:", float((L@new_dirs.T).min()), float((L@new_dirs.T).max()))

for ex in ("action","video"):
    for suffix in ("text_embedding.0.bias","text_embedding.2.bias","text_embedding.2.weight"):
        n = f"mot.mixtures.{ex}.{suffix}"
        dif = load(n) - load_original(n)
        if dif.ndim == 1:
            idx = torch.where(dif.abs()>1e-6)[0]
            print(ex, suffix, "changed", idx.tolist(),
                  "current", load(n)[idx].tolist())
        else:
            rn = dif.norm(dim=1); cn = dif.norm(dim=0)
            print(ex, suffix, "largest row", int(rn.argmax()),
                  "largest columns", torch.topk(cn, 6).indices.tolist(),
                  "nonzero rows/cols", int((rn>1e-6).sum()), int((cn>1e-6).sum()))
with (W/"notes.md").open("a") as f:
    f.write("\nPass30 rank320 hybrid raw development: 48->51/96, "
            "+3.125pp SE3.8, 8gains/5losses. No improvement over smaller rank256; "
            "rank320 not selected.\n")
```
