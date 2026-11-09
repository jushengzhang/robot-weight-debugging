The masked-prefix contrasts have a compact, well-conditioned spectrum: about 99% of their protected-complement energy lies in 194 directions. I will test a rank-192 prefix-focused projection with the existing gate. This removes fewer directions overall while targeting the corruption that remains after high-confidence suffix suppression.
```python
more_helpers=r'''
def install_hybrid_basis(basis_file,rank,dose=1.,experts=("action","video")):
    basis=torch.load(W/basis_file,weights_only=False,map_location="cpu")
    u=basis["U"][:,:rank].double();b=basis["B_nearest"][:,:rank].double()
    snap=W/"candidates/hybrid_r256_d1_t05_g128"
    for ex,h in [("action",296),("video",2844)]:
        n=f"mot.mixtures.{ex}.text_embedding.0.weight";o=load_original(n).double()
        d=dose if ex in experts else 0.
        a=(o-d*(o@b)@u.T).float()
        a[h]=torch.from_numpy(np.load(snap/(n+".npy")))[h]
        save_edit(n,a)
    cfg={"kind":"hybrid_oblique_suffix_gate","basis_file":basis_file,"rank":rank,"dose":dose,
         "threshold":.5,"gain":128.,"strength":6.,"experts":list(experts)}
    (W/"current_candidate.json").write_text(json.dumps(cfg,indent=2))
    return cfg
'''
with (W/"analysis_helpers.py").open("a") as f:f.write("\n"+more_helpers)
exec(more_helpers)
install_hybrid_basis("masked_prefix_projection.pt",192,1.)
with (W/"notes.md").open("a") as f:
    f.write("\nPass27 planned hybrid PREFIX192: SVD only15 suffix-masked context queries per raw input, same nearest-clean targets and521D protection. 99% protected-complement contrast energy rank194, so choose192 (basis RMS1.868). Prefix contrast residual norm.288 vs.345 for prior all-queryrank256. Avoid rank256 prefix due tiny singular values/oblique amplification8x. Gate unchanged threshold.5/gain128/strength6, both experts. Analytic subspace choice, no action-target fitting.\n")
print((W/"current_candidate.json").read_text())
```
