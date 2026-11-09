I will test a conservative late-MLP candidate: one contrast direction each in layers 11, 19, and 25. These are the later layers with task-split overlap above 0.70 and failure/control variance ratio above 100. The edit will halve those centered outputs while preserving the control mean. I will also prepare a norm-matched, texture-orthogonal control.
```python
late_layers = [
    r["layer"] for r in mstats
    if r["direction"] == 0 and r["layer"] >= 10
    and r["overlap"] >= .70 and r["failure_control_ratio"] >= 100
]
assert late_layers == [11, 19, 25], late_layers
cons = np.load(W / "failure_mlp_consensus.npz")
rng = np.random.default_rng(31031)
late_data = {}

for L in late_layers:
    u = torch.from_numpy(cons[f"U{L}"][:, 0].copy()).float()
    u /= u.norm()
    q = torch.from_numpy(cons[f"Q{L}"].copy()).float()
    allu = torch.from_numpy(cons[f"U{L}"].copy()).float()
    protect = torch.linalg.qr(torch.cat([q, allu], dim=1), mode="reduced").Q
    c = torch.from_numpy(rng.standard_normal(1024).astype(np.float32))
    c -= protect @ (protect.T @ c)
    c /= c.norm()
    wn = f"mot.mixtures.action.blocks.{L}.ffn.2.weight"
    ow = load_original(wn)
    scale = (u @ ow).norm() / (c @ ow).norm()
    late_data.update({
        f"target{L}": u.numpy(), f"control{L}": c.numpy(),
        f"mean{L}": cons[f"mean{L}"].copy(),
        f"control_scale{L}": np.array(float(scale)),
    })
np.savez(W / "late_mlp_causal_directions.npz", **late_data)

def set_late_mlp_edit(dose, kind="target"):
    for name in list(edited()):
        drop_edit(name)
    changes = []
    for L in late_layers:
        u = torch.from_numpy(late_data[f"{kind}{L}"])
        mu = torch.from_numpy(late_data[f"mean{L}"])
        a = dose * (float(late_data[f"control_scale{L}"]) if kind == "control" else 1.0)
        wn = f"mot.mixtures.action.blocks.{L}.ffn.2.weight"
        bn = f"mot.mixtures.action.blocks.{L}.ffn.2.bias"
        ow, ob = load_original(wn), load_original(bn)
        ew = ow - a * torch.outer(u, u @ ow)
        eb = ob + a * u * torch.dot(u, mu - ob)
        if dose:
            save_edit(wn, ew)
            save_edit(bn, eb)
        changes.append({
            "layer": L, "actual_dose": a,
            "weight_relative_change": float((ew-ow).norm()/ow.norm()),
            "bias_change_norm": float((eb-ob).norm()),
        })
    spec = {"family":"protected_failure_specific_late_MLP", "dose":dose,
            "kind":kind, "rank_per_layer":1, "changes":changes}
    (W / "current_candidate.json").write_text(json.dumps(spec, indent=2))
    return spec

print("Candidate:", set_late_mlp_edit(.5))
print("WORK:", str(W))
fm = readj("fixed_residual_probe_manifest.json")
print("Manifest keys:", list(fm))
for k,v in fm.items():
    if isinstance(v, list):
        print(k, len(v), v[:2])
print("Short input paths:", str(readj("causal_probe_paths.json"))[:1300])
print("i directory example:", [p.name for p in sorted((W/"i").iterdir())[:10]])
with (W / "notes.md").open("a") as f:
    f.write(
        "\n## Final within-family late-MLP test\n"
        "- Early MLP suppression did not propagate to actions. Existing fixed-input data identify "
        "L11/L19/L25 as late leading directions with task-half overlap>=.70 and failure/control "
        "variance ratio>=100. Rank1 centered output projections only; first dose .5.\n"
        "- Plan: fixed-input .5 and1.0 doses, orthogonal norm-matched control, then locked-candidate "
        "disjoint paired rollout validation. No further candidate families. Reject if causal specificity "
        "or validation gain is absent. Original weights remain the fallback.\n"
    )
```
