The extreme position row at patch 193 does coincide with a fixed base-camera hotspot, but the wrist hotspot does not—consistent with the wrist-head ablation failing. A separate upstream candidate is the projector's leading mode: it strongly feeds the shared outlier channels and varies little across scenes. I will attenuate only that rank-one mode by 20%, preserving all other visual projection directions, and test whether visual detail gains relative influence.
```python
clear_edits()
P0=P.clone()
projector_rank1=sP[0]*torch.outer(uP[:,0],vhP[0])
save_edit(proj_name,P0-0.2*projector_rank1)
snapshot("candidate_projector_top1_keep080")
print("Relative matrix edit:",round(float((load(proj_name)-P0).norm()/P0.norm()),5))
print("Analytic retained mode ratio:",
      round(float(uP[:,0]@load(proj_name)@vhP[0]/sP[0]),5))
# Prepare, but do not install, an independent position-register intervention for a later causal test.
pe0=load_original(pe_name)
outlier_positions=(pe0.norm(dim=1)>3*pe0.norm(dim=1).median()).nonzero().flatten().tolist()
pe_reference=pe0.clone()
for j in outlier_positions:
    rr,cc=divmod(j,16)
    near=[r*16+c for r in range(max(0,rr-2),min(16,rr+3))
                   for c in range(max(0,cc-2),min(16,cc+3))
                   if r*16+c not in outlier_positions]
    pe_reference[j]=pe0[near].mean(0)
print("Position outliers reserved for separate test:",outlier_positions)
with (W/"notes.md").open("a") as f:
    f.write("\n## Weight-localized upstream candidates\n"
            "Projector top singular value15.084 vs second3.689; top mode12.88% of matrix energy, "
            "left vector dominated by dims674(0.733),604(0.279),738(0.172). "
            "Pooled visual input score mean-67.10, SD2.11 across128 scenes, largely common-mode. "
            "Candidate E: projector W <- W -0.2*s0*u0*v0^T (only top singular mode attenuated).\n"
            "Position embedding outlier rows193(norm313.8),47(231.1),48(198.1),243(97.3),13(93.9), "
            "median18.24. Patch193 is a recurring base-camera attention hotspot; wrist211 is NOT a positional outlier. "
            "Prepare independent outlier-residual attenuation preserving local positional neighborhood; not installed yet.\n")
```
