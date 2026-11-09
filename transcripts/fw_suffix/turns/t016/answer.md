The extra entries are per-denoising-step attention outputs, which lets me examine the first step before noise trajectories diverge. I will use the clean-protected contrast directions for a rank-4 projection in the action expert’s shared-attention output projections, leaving its text cross-attention and all video weights unchanged for this first causal test.
```python
print("First denoising step: raw-clean relative attention-output difference")
for l in range(30):
    key=f"calls::action.{l}.attn_out::action"
    R=np.stack([probe4_acts[2*i][key][0] for i in range(24)])
    C=np.stack([probe4_acts[2*i+1][key][0] for i in range(24)])
    if l in [0,1,2,3,4,5,7,10,12,15,20,23,26,29]:
        print(l,round(float(np.linalg.norm(R-C)/(np.linalg.norm(C)+1e-12)),5))

def project_output(expert, module, layers, bases, alpha=1.0, rank=4):
    total_sq=0.0
    delta_sq=0.0
    for l in layers:
        U=bases[(expert,l)][0][:,:rank]
        for tail in ["weight","bias"]:
            n=f"mot.mixtures.{expert}.blocks.{l}.{module}.o.{tail}"
            W=load_original(n)
            delta=alpha*(U@(U.T@W))
            save_edit(n, W-delta)
            total_sq+=float(W.square().sum())
            delta_sq+=float(delta.square().sum())
    return math.sqrt(delta_sq/total_sq)

clear_edits()
action_projection_layers = list(range(1,30))
edit_norm = project_output("action","self_attn",action_projection_layers,basis_data,alpha=1.0,rank=4)
config={"name":"action_self_P4","module":"action.self_attn.o",
        "layers":action_projection_layers,"rank":4,"alpha":1.0,"protected":"clean mean span"}
(work/"current_edit.json").write_text(json.dumps(config,indent=2))
print("Current edit:", len(edited()),"tensors; relative parameter delta norm:",round(edit_norm,5))
with (work/"notes.md").open("a") as f:
    f.write("\n### Candidate action_self_P4\n"
            "For action layers 1-29, left-project self_attn.o weight and bias with "
            "(I-UU^T), U=top 4 tagged-clean difference directions after removing the "
            "span of 24 clean mean outputs (from probe 4). Static SVD/projection, no training. "
            f"Relative changed-parameter norm={edit_norm:.6f}. Probe pending.\n")
```
