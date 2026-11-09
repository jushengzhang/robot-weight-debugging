The matched background-versus-light comparison reveals a much more reproducible shift: at the final action residual, its split-half cosine is 0.882, and its projection is about 20 times larger on background contrasts than on light-task variation. I’ll test a rank-one removal of this direction from the action readout at 50% strength. This leaves the visual encoder and attention weights unchanged and protects the light mean and 95% of observed light-task variation.
```python
L = 29
u = torch.from_numpy(mean_shift_data["u29"].copy())
Q = torch.from_numpy(mean_shift_data["Q29"].copy())
mu = torch.from_numpy(mean_shift_data["mean29"].copy())
H0 = load_original("mot.mixtures.action.head.weight")
dose = 0.5
H1 = H0 - dose * torch.outer(H0@u, u)
assert torch.isfinite(H1).all()
assert float((H1@Q-H0@Q).norm()) < 1e-5
assert not edited()
save_edit("mot.mixtures.action.head.weight", H1)

candidate = {
    "family":"task_protected_texture_mean_readout",
    "dose":dose, "layer_source":29, "rank":1,
    "tensor":"mot.mixtures.action.head.weight",
    "basis_file":"matched_mean_shift_subspaces.npz",
    "basis_key":"u29",
    "relative_weight_change":float((H1-H0).norm()/H0.norm()),
    "mean_preservation_error":float((H1@mu-H0@mu).norm()),
    "protected_basis_error":float((H1@Q-H0@Q).norm()),
    "readout_of_nuisance":(H0@u).tolist(),
    "diagnostic_stats":mean_shift_stats[29]
}
(W / "current_candidate.json").write_text(json.dumps(candidate,indent=2))
snapshot = W / "candidates" / "readout_mean_d050"
snapshot.mkdir(exist_ok=True)
np.save(snapshot / "head.weight.npy", H1.numpy())
(snapshot / "candidate.json").write_text(json.dumps(candidate,indent=2))

# Construct a norm-matched negative-control direction, orthogonal to observed
# texture contrasts, protected light directions, and the target direction.
texture_d = ((Xraw[T[:,0]]+Xraw[T[:,1]])/2-Xraw[T[:,2]])
P = torch.linalg.qr(torch.cat([Q,texture_d.T,u[:,None]],1),mode="reduced").Q
Hfree = H0-(H0@P)@P.T
_, sh, vh = torch.linalg.svd(Hfree,full_matrices=False)
e1 = vh[0]
g = torch.Generator().manual_seed(80413)
e0 = torch.randn(1024,generator=g)
S = torch.linalg.qr(torch.cat([P,vh.T],1),mode="reduced").Q
e0 = e0-S@(S.T@e0)
e0 = e0/e0.norm()
mix = float((H0@u).norm()/sh[0])
assert 0 <= mix <= 1, mix
control_u = mix*e1+math.sqrt(1-mix*mix)*e0
np.savez_compressed(W / "readout_causal_directions.npz",
                    target=u.numpy(),control=control_u.numpy())
print("Candidate:",candidate)
print("Control check:",{
    "norm":float(control_u.norm()),
    "delta_weight_norm_ratio":float((H0@control_u).norm()/(H0@u).norm()),
    "protected_overlap":float((Q.T@control_u).norm()),
    "texture_overlap_rms":float((texture_d@control_u).square().mean().sqrt())
})
with (W / "notes.md").open("a") as f:
    f.write("\n## Task-protected texture mean-shift hypothesis\n")
    f.write("- High-vs-low texture variance directions failed stability screen; not edited.\n")
    f.write("- Matched (high+low)/2 minus light residual mean is coherent across all27 matched tasks, with final-layer split cosine .8818 and texture/light projection RMS 3.4147/.1705. Light PCA25 protects95.4% of task variation; mean and constant also protected.\n")
    f.write("- Experimental rank1 final action readout edit H'=H-.5(Hu)u^T, no bias or other changes. This targets a texture-linked internal feature, not global action scaling. Norm-matched texture-orthogonal control prepared.\n")
```
