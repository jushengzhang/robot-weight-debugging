I will construct a single protected contrast direction at layer 11: remove the span of the successful reference outputs from the paired instruction differences, then take their leading principal direction. The proposed edit will attenuate only that direction in the action expert’s shared-attention output projection, leaving the protected reference means unchanged algebraically.
```python
pr8=json.loads((W/"probe_8/results.json").read_text())
sel_by_path={r["input"]:r for r in selected}
sel8=[sel_by_path[str(Path(r["input"]).resolve())] for r in pr8]
L8=[0,8,9,10,11,12,13,14,15,16,17,18,20,23,27,29]
A8={l:torch.from_numpy(np.stack([
    np.load(r["activations"])[f"mean::action.{l}.attn_out::action"] for r in pr8])).float()
    for l in L8}
good8=torch.tensor([r["success"] for r in sel8],dtype=torch.bool)
pair_indices=[([i for i,r in enumerate(pr8) if r["input"]==pr4[2*j]["input"]][0],
               [i for i,r in enumerate(pr8) if r["input"]==pr4[2*j+1]["input"]][0])
              for j in range(10)]
protected_modes={}
print("Layer retained contrast energy, PC1 fraction, reference leakage:")
for l in L8:
    T=A8[l]
    ref=T[good8]
    _,sr,vhr=torch.linalg.svd(ref,full_matrices=False)
    Q=vhr[sr>sr.max()*1e-5].T
    diff=torch.stack([T[b]-T[g] for g,b in pair_indices])
    dn=diff/diff.norm(dim=1,keepdim=True).clamp_min(1e-8)
    dp=dn-(dn@Q)@Q.T
    _,s,vh=torch.linalg.svd(dp,full_matrices=False)
    u=vh[0]; u=u-Q@(Q.T@u);u=u/u.norm()
    protected_modes[l]={"u":u,"s":s,"Q":Q,
                        "retained_frac":float(dp.square().sum()/dn.square().sum())}
    print(l,round(protected_modes[l]["retained_frac"],4),
          round(float(s[0]**2/s.square().sum()),4),
          f"{float((ref@u).abs().max()):.2e}")
torch.save(protected_modes,W/"protected_instruction_contrast_modes.pt")

def set_direction_edit(layer, module, alpha, label, direction=None):
    clear_edits()
    u=(protected_modes[layer]["u"] if direction is None else direction).float()
    changes=[]
    for suffix in ["weight","bias"]:
        n=f"mot.mixtures.action.blocks.{layer}.{module}.o.{suffix}"
        t=load_original(n).clone()
        delta=(-alpha*torch.outer(u,u@t) if suffix=="weight"
               else -alpha*u*torch.dot(u,t))
        save_edit(n,t+delta)
        changes.append({"name":n,"relative_delta":float(delta.norm()/t.norm())})
    meta={"label":label,"layer":layer,"module":module,"alpha":alpha,"rank":1,
          "changes":changes}
    (W/"active_candidate.json").write_text(json.dumps(meta,indent=2))
    return meta

print("\nCandidate:",set_direction_edit(11,"self_attn",0.5,"L11_self_protected_PC1_half"))
print("Projection sign-free scores for 10 paired differences:",
      [round(float((A8[11][b]-A8[11][g])@protected_modes[11]["u"]),3) for g,b in pair_indices])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 8 and protected projection candidate\n"
            "- Original attention-output probes on 48 registered saved inputs; 37 successful references cover 36 tasks. Refused convenience directory deleted; actual probe used only original file paths.\n"
            "- Closed-form candidate: remove successful-output span from the ten paired instruction differences, normalize pair weights, take leading PCA direction u at L11. Modify action L11 self_attn.o weight and bias by (I - 0.5 u u^T). No optimizer, loss, gradients, or policy execution in kernel. This protects reference mean outputs algebraically, not all tokens or future states; must test behavior.\n")
```
