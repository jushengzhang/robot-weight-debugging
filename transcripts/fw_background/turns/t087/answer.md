The saved analysis suggests a different route: some heads transmit the texture-sensitive video feature through their **values**, even if their attention maps do not look unusually background-focused. I’ll estimate that transfer and look for a narrow action-head value direction that can be removed while protecting the light-control feature subspace and leaving the video encoder untouched.
```python
print("Relevant projection names:")
for n in NAMES:
    if "blocks.10.self_attn" in n:
        print(n, SHAPES[n])

vd = np.load(W / "video_value_nuisance_data.npz")
ad = np.load(W / "projection_diagnostic_data.npz")
print("Action contrast archive keys:", ad.files[:10])
print("Video protected basis checks:",
      [(L, round(float(np.linalg.norm(vd[f"Q{L}"].T @ vd[f"U{L}"][:,0])),6))
       for L in [10,14,20,25,29]])

# Rank a possible action-only value-direction intervention. No edits are made here.
ranked = []
direction_data = {}
for L in [10,14,20,25,29]:
    prefix_v = f"mot.mixtures.video.blocks.{L}.self_attn."
    prefix_a = f"mot.mixtures.action.blocks.{L}.self_attn."
    vname = prefix_v + "v.weight"
    oname = prefix_a + "o.weight"
    if oname not in NAMES:
        oname = next(n for n in NAMES if n.startswith(prefix_a) and SHAPES[n] == (1024,3072))
    V = load_original(vname)
    O = load_original(oname)
    u = torch.from_numpy(vd[f"U{L}"][:,0].copy())
    Q = torch.from_numpy(vd[f"Q{L}"].copy())
    projected_u = (V @ u).reshape(24,128)
    protected = (V @ Q).reshape(24,128,-1)
    AU = torch.from_numpy(ad[f"U{L}"][:,:2].copy())
    map_mass = None
    if str(L) in p17[0].get("attention", {}):
        map_mass = np.mean([np.load(r["attention"][str(L)]).sum(-1).mean(-1) for r in p17],axis=0)
    for h in range(24):
        qv = torch.linalg.qr(protected[h],mode="reduced").Q
        b0 = projected_u[h]
        b = b0 - qv @ (qv.T @ b0)
        bnorm = float(b.norm())
        if bnorm < 1e-8:
            continue
        b = b / bnorm
        ob = O[:,h*128:(h+1)*128] @ b
        transfer = float(ob.norm()) * bnorm
        contrast_alignment = float((AU.T @ ob).norm() / (ob.norm()+1e-12))
        mass = float(map_mass[h]) if map_mass is not None else None
        score = transfer * contrast_alignment * (mass if mass is not None else 1.0)
        ranked.append({"layer":L, "head":h, "transfer":transfer,
                       "contrast_alignment":contrast_alignment, "video_mass":mass,
                       "retained_nuisance":bnorm/(float(b0.norm())+1e-12), "score":score})
        direction_data[f"L{L}H{h}"] = b.numpy()
    del V, O

ranked.sort(key=lambda r:r["score"],reverse=True)
print("\nTop protected value-transfer candidates:")
for r in ranked[:20]:
    print({k:round(v,5) if isinstance(v,float) else v for k,v in r.items()})

np.savez_compressed(W / "action_value_direction_candidates.npz", **direction_data)
(W / "action_value_direction_ranking.json").write_text(json.dumps(ranked,indent=2))
print("No edits:", edited())
```
