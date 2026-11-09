The patch edit fails the rollout test: background success drops by 3.1 points and light success by 5.4 points. I’ll revert it. For the final candidate, I’ll test the two early MLP directions that passed the failure-specific reproducibility screen, using a centered rank-one edit that preserves the control means. The remaining budget is reserved for its causal tests and independent validation, not further hypothesis searching.
```python
rows28 = paired_rows(28)
s28 = {c:paired_summary([r for r in rows28 if r["category"]==c])
       for c in sorted({r["category"] for r in rows28})}
(W/"roll28_summary.json").write_text(json.dumps(s28,indent=2))
for n in list(edited()):
    drop_edit(n)
print("Rejected patch result:",s28)

mlp_causal = {}
for L in [1,2]:
    direction = torch.from_numpy(mlp_consensus_data[f"U{L}"][:,0].copy())
    q = torch.from_numpy(mlp_consensus_data[f"Q{L}"].copy())
    muL = torch.from_numpy(mlp_consensus_data[f"mean{L}"].copy())
    dl = torch.from_numpy(mlp_data[f"D{L}"])
    constraints = torch.linalg.qr(torch.cat([q,dl.T,direction[:,None]],1),mode="reduced").Q
    g = torch.Generator().manual_seed(90200+L)
    ctl = torch.randn(1024,generator=g)
    ctl = ctl-constraints@(constraints.T@ctl)
    ctl = ctl/ctl.norm()
    wn = f"mot.mixtures.action.blocks.{L}.ffn.2.weight"
    bn = f"mot.mixtures.action.blocks.{L}.ffn.2.bias"
    w0,b0 = load_original(wn),load_original(bn)
    aug = torch.cat([w0,(b0-muL)[:,None]],1)
    control_scale = float((direction@aug).norm()/(ctl@aug).norm())
    mlp_causal[f"target{L}"] = direction.numpy()
    mlp_causal[f"control{L}"] = ctl.numpy()
    mlp_causal[f"mean{L}"] = muL.numpy()
    mlp_causal[f"control_scale{L}"] = np.array(control_scale)

np.savez_compressed(W/"early_mlp_causal_directions.npz",**mlp_causal)
mlp_utility = '''
def set_early_mlp_edit(dose, kind="target"):
    W = Path(WORK)
    for name in list(edited()):
        drop_edit(name)
    z = np.load(W/"early_mlp_causal_directions.npz")
    changes = []
    for L in [1,2]:
        u = torch.from_numpy(z[f"{kind}{L}"].copy())
        mu = torch.from_numpy(z[f"mean{L}"].copy())
        a = dose*(float(z[f"control_scale{L}"]) if kind=="control" else 1.0)
        wn = f"mot.mixtures.action.blocks.{L}.ffn.2.weight"
        bn = f"mot.mixtures.action.blocks.{L}.ffn.2.bias"
        w,b = load_original(wn),load_original(bn)
        we = w-a*torch.outer(u,u@w)
        be = b+a*u*torch.dot(u,mu-b)
        if dose:
            save_edit(wn,we)
            save_edit(bn,be)
        changes.append({"layer":L,"actual_dose":a,
                        "weight_relative_change":float((we-w).norm()/w.norm()),
                        "bias_change_norm":float((be-b).norm())})
    spec = {"family":"protected_failure_specific_early_MLP",
            "dose":dose,"kind":kind,"rank_per_layer":1,"changes":changes}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
    return spec
'''
with (W/"experiment_utils.py").open("a") as f:
    f.write(mlp_utility)
exec(mlp_utility)
print("Applied:",set_early_mlp_edit(.5))
with (W/"notes.md").open("a") as f:
    f.write("- Pass28 patchDC dose .25: "+json.dumps(s28)+". Rejected for background and light regressions; patch edit removed, stronger dose/control skipped for futility.\n")
    f.write("\n## Final exploratory candidate\n")
    f.write("- Action MLP output projections L1/L2 only; leading failure-specific shared subspaces have task-half overlaps .932/.843 and failure/control ratios11.56/14.43. Controls protect light and successful-background means and95% variation.\n")
    f.write("- W'=W-.5u(u^TW), b'=b+.5u[u^T(mu-b)]. Rank1 each, centered. Norm-matched texture-orthogonal controls prepared. No further new families after this; reserve independent validation.\n")
```
