The failed ablation barely changed the gripper output on identical inputs; its main effect was on translation. I will therefore stop using raw readout alignment as a proxy for gripper causality. The next test targets visual routing directly: remove one camera-contrast direction from the query projection of layer 10's spatial head, then check whether attention actually shifts from wrist-dominated context toward the base view.
```python
# Construct a direction from recorded group means, without fitting any predictor or running inference.
# Limit to slow RoPE pairs so that the camera contrast is not dominated by rapidly rotating coordinates.
camera_dirs={}
lowrope=torch.zeros(256)
lowrope[96:128]=1
lowrope[224:256]=1
for l in [8,10,12,13]:
    D=torch.stack([torch.from_numpy(z[f"mean::vlm.{l-1}.resid::wrist"]-
                                    z[f"mean::vlm.{l-1}.resid::base"]) for z in res4])
    d=D.mean(0)
    K=load_original(f"{VL}.layers.{l}.self_attn.k_proj.weight")
    kd=K@d
    kd=kd*lowrope
    kd=kd/kd.norm()
    camera_dirs[l]=kd
    proj=(D@K.T)@kd
    print("Layer",l,"camera contrast sign consistency",float((proj>0).float().mean()),
          "mean/std",round(float(proj.mean()),3),round(float(proj.std()),3))
def camera_query_edit(layer,head,strength):
    n=f"{EX}.layers.{layer}.self_attn.q_proj.weight"
    T=load_original(n).clone()
    q=T[head*256:(head+1)*256]
    d=camera_dirs[layer]
    delta=d[:,None]*(d@q)[None,:]
    q-=strength*delta
    save_edit(n,T)
    return n
clear_edits()
cn=camera_query_edit(10,1,1.0)
print("Edit:",cn,"head 1 rank-one camera-direction removal")
print("Relative tensor delta:",round(float((load(cn)-load_original(cn)).norm()/load_original(cn).norm()),5))
torch.save(camera_dirs,W/"camera_contrast_directions.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 6 interpretation and next candidate\n"
            "- E16H3 half-output: identical-input mean gripper differences only ~0.001, "
            "translation RMS changes 0.004-0.025. Raw output-weight alignment was not causal gripper evidence.\n"
            "- Next: rank-one removal of a recorded wrist-minus-base key contrast from L10 H1 query, "
            "restricted to slow RoPE coordinate pairs. Probe must confirm the intended attention change.\n")
```
