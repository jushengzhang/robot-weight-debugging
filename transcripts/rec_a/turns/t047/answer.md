The wrist contrast is reasonably consistent across the matched scenes, but the latest layers also carry substantial shared content. I will remove the common base-image, wrist-image, and instruction components from the candidate direction before editing it, to avoid simply suppressing generic visual information.
```python
centered_dirs = {}
print("Layer: top value singular-value share; contrast retained after common-content protection; projected O effect")
for layer in range(1,18):
    l = layer - 1
    scale = 1 + load_original(V + f"layers.{layer}.input_layernorm.weight")
    wv = load_original(V + f"layers.{layer}.self_attn.v_proj.weight")
    means = []
    for region in ("base","wrist","lang"):
        x = torch.from_numpy(np.stack([a[f"mean::vlm.{l}.resid::{region}"] for a in a15]).mean(0))
        means.append(wv @ (x*scale))
    M = torch.stack(means, dim=1)
    U,S,_ = torch.linalg.svd(M, full_matrices=False)
    basis = U[:, S > S[0]*1e-5]
    vd0 = visdirs[layer]["value_direction"]
    vd = vd0 - basis @ (basis.T @ vd0)
    retained = float(vd.norm())
    vd /= vd.norm()
    centered_dirs[layer] = vd
    # No submodel execution: spectral and linear weight geometry only.
    vals = torch.linalg.svdvals(wv)
    o = load_original(E + f"layers.{layer}.self_attn.o_proj.weight")
    shifts = torch.stack([o[:,h*256:(h+1)*256] @ vd for h in range(8)],dim=1)
    gshift = g @ shifts
    print(layer, round(float(vals[0]**2/(vals**2).sum()),3),round(retained,3),
          round(float(shifts.norm()),3),round(float(gshift.norm()),3))
torch.save(centered_dirs, W/"visual_gripper_value_dirs_centered.pt")

# Initial causal test: halve the protected gripper-state value direction in
# middle expert layers, where the matched visual contrast remains consistent.
candidate_layers = list(range(1,15))
alpha = 0.5
for n in list(edited()):
    drop_edit(n)
stats = []
for layer in candidate_layers:
    n = E + f"layers.{layer}.self_attn.o_proj.weight"
    w0 = load_original(n)
    w = w0.clone()
    vd = centered_dirs[layer]
    for h in range(8):
        sl = slice(256*h,256*(h+1))
        w[:,sl] -= alpha * torch.outer(w0[:,sl] @ vd, vd)
    save_edit(n,w)
    stats.append(float((w-w0).norm()/w0.norm()))
spec = {"name":"visual_grip_value_mid_alpha0.5", "layers":candidate_layers,
        "alpha":alpha, "mechanism":"attenuate matched visual gripper-state contrast in expert value readout; protect three modality means",
        "directions":"visual_gripper_value_dirs_centered.pt"}
(W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
print("Candidate:",spec["name"],"tensors",len(edited()),"relative changes range",min(stats),max(stats))
with (W/"notes.md").open("a") as f:
    f.write("- Matched wrist residual contrast mean cosine across scenes ~0.50-0.57 in layers 0-13; two-half value-direction cosine ~0.72-0.83.\n"
            "- Candidate: expert O projections at L1-14 attenuate only the corresponding one-dimensional visual gripper-state value direction by 50%, across heads. Directions orthogonalized to mean base/wrist/language values. No output/action/noise scaling.\n")
```
