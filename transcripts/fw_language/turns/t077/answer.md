The stronger attention-softening dose hurt spatial tasks (-3.1 +/- 1.9 points), so I will revert it. The earlier protected projection preserved task features but removed only the small off-task component of the wording contrast. I will test a rank-one oblique projection instead: use that protected component to identify the contrast, but attenuate the full corresponding wording direction. This is a closed-form geometric edit, with no loss fitting or optimization.
```python
clear_edits()
l=11
T=A8[l]
ref=T[good8]
Q=protected_modes[l]["Q"]
diff=torch.stack([T[b]-T[g] for g,b in pair_indices])
dn=diff/diff.norm(dim=1,keepdim=True).clamp_min(1e-8)
dp=dn-(dn@Q)@Q.T
left,s,right=torch.linalg.svd(dp,full_matrices=False)
u=right[0]
u=u-Q@(Q.T@u);u=u/u.norm()
d=dn.T@left[:,0]
d=d/torch.dot(u,d)  # u^T d = 1; one oblique spectral projector.
oblique={"layer":11,"read_direction":u,"contrast_direction":d,
         "singular_values":s,"reference_rank":Q.shape[1]}
torch.save(oblique,W/"L11_protected_oblique_projection.pt")
print("Projector geometry:",
      {"u_dot_d":round(float(u@d),7),"d_norm":round(float(d.norm()),4),
       "max_reference_read":float((ref@u).abs().max())})
print("Algebraic contrast RMS original/projected half:",
      round(float(diff.square().mean().sqrt()),5),
      round(float((diff-0.5*(diff@u)[:,None]*d[None,:]).square().mean().sqrt()),5))
def set_oblique_edit(alpha):
    clear_edits()
    changes=[]
    for suffix in ["weight","bias"]:
        n=f"mot.mixtures.action.blocks.11.self_attn.o.{suffix}"
        t=load_original(n)
        delta=(-alpha*torch.outer(d,u@t) if suffix=="weight" else -alpha*d*(u@t))
        save_edit(n,t+delta)
        changes.append({"name":n,"relative_delta":float(delta.norm()/t.norm())})
    meta={"label":f"L11_protected_oblique_alpha{alpha}","alpha":alpha,
          "rank":1,"changes":changes}
    (W/"active_candidate.json").write_text(json.dumps(meta,indent=2))
    return meta
print(set_oblique_edit(0.5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 15 and geometric follow-up\n"
            "- H19 query scale .50: 90/128 original vs 86/128 edited; -3.1 pp +/-1.9 SE (1 rescue, 5 regressions, including one error). Reverted. Excessive softening harms spatial precision; .75 result not accepted.\n"
            "- Follow-up to pass 9: orthogonal deflation removes only the task-orthogonal component of a wording contrast, leaving its task-overlapping component. Construct rank-one oblique projection from the same normalized paired differences: u = protected contrast PC, d = corresponding full contrast with u^T d=1. Edit action L11 self-attention O and bias by (I-alpha*d*u^T), alpha=.5. Successful reference mean outputs are algebraically preserved. Closed-form PCA/projector, no regression targets, loss fitting, or optimization. Awaiting causal probe.\n")
```
