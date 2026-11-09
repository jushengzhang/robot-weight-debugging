Layers 4 and 5 offer a more selective target: their leading contrast directions are substantially more variable under texture changes than across light controls, while the full rank-4 subspaces contain only 8.4% and 13.9% of light-control variation. I will attenuate the leading two directions by 35%, with bias compensation, and test whether the predicted action sensitivity actually falls.
```python
def apply_projection(layers=(4,5), rank=2, dose=0.35, control=False):
    clear_edits()
    report=[]
    rng=np.random.default_rng(83291)
    for L in layers:
        U=torch.from_numpy(projection_data[f"U{L}"][:,:rank].copy()).float()
        mu=torch.from_numpy(projection_data[f"mean{L}"].copy()).float()
        name=f"mot.mixtures.action.blocks.{L}.self_attn.o"
        wo=load_original(name+".weight"); bo=load_original(name+".bias")
        if control:
            # Fixed random directions orthogonal to the full measured contrast basis.
            R=torch.from_numpy(rng.standard_normal((1024,rank)).astype(np.float32))
            V=torch.from_numpy(projection_data[f"U{L}"]).float()
            R=R-V@(V.T@R)
            U=torch.linalg.qr(R, mode="reduced").Q
        dw=-dose*U@(U.T@wo)
        db=dose*U@(U.T@(mu-bo))
        save_edit(name+".weight",wo+dw)
        save_edit(name+".bias",bo+db)
        report.append({"layer":L,"rank":rank,"dose":dose,
                       "weight_relative_change":float(dw.norm()/wo.norm()),
                       "bias_delta_rms":float(db.square().mean().sqrt()),
                       "anchor_error":float((dw@torch.zeros(wo.shape[1])).abs().max())})
    spec={"family":"centered_output_subspace_attenuation","layers":list(layers),
          "rank":rank,"dose":dose,"control":control,"tensors":report}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
    return spec
candidate=apply_projection()
print(json.dumps(candidate,indent=2))
# Preserve the analytically computed candidate independently of evaluation results.
snap=W/"candidates"/"L4L5_rank2_d035"; snap.mkdir(parents=True,exist_ok=True)
for n in edited():
    torch.save(load(n),snap/(n+".pt"))
(snap/"spec.json").write_text(json.dumps(candidate,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 7 and spectral intervention design\n"
            "- 37 original light initial-observation references obtained.\n"
            "- L4 leading contrast/light variance ratios 24.45,16.15; L5 14.13,22.79. Full rank-4 subspaces contain 8.4%,13.9% of light task variation. Leading subspaces reproduce across pair halves.\n"
            "- Candidate: action self-attention output L4/L5, rank-2 centered projection, dose .35. W' = W - a U U^T W; b' = b + a U U^T(mu_light-b). No training, video changes or global scaling.\n"
            "- Experimental, not accepted. Passes used 7/40.\n")
```