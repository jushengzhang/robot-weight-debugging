The late gripper-readout intervention also failed to improve net success, so I will restore it. The remaining evidence points more toward which visual evidence reaches the expert than toward the gripper output itself. I will test whether a camera-specific offset in the late VLM keys biases attention toward the local wrist view, using a rank-one key edit derived from camera-group means rather than success labels.
```python
reset_edits()
def camera_direction(prev_layer):
    xb=torch.from_numpy(np.stack([a[f"mean::vlm.{prev_layer}.resid::base"] for a in AR7]))
    xw=torch.from_numpy(np.stack([a[f"mean::vlm.{prev_layer}.resid::wrist"] for a in AR7]))
    xl=torch.from_numpy(np.stack([a[f"mean::vlm.{prev_layer}.resid::lang"] for a in AR7]))
    nw=load_original(f"{VL}.layers.{prev_layer+1}.input_layernorm.weight")
    # Geometry of token-group means in the input layernorm's learned coordinate system.
    # This is a summary-statistic direction, not reconstructed per-token network inference.
    def norm_summary(x):
        return x/(x.square().mean(dim=1,keepdim=True)+1e-6).sqrt()*(1+nw)
    b,w,l=map(norm_summary,[xb,xw,xl])
    common=(b.mean(0)+w.mean(0))/2
    basis=torch.stack([common,l.mean(0)],1)
    qq,_=torch.linalg.qr(basis,mode="reduced")
    d=(w-b).mean(0)
    d=d-qq@(qq.T@d)
    d=d/d.norm()
    consistency=torch.nn.functional.cosine_similarity(w-b,d[None],dim=1)
    print("camera axis",prev_layer+1,"norm weight min/mean/max",
          [round(float(f(nw)),3) for f in [torch.min,torch.mean,torch.max]],
          "per-observation cosine mean/min",round(float(consistency.mean()),3),round(float(consistency.min()),3))
    return d
cam14=camera_direction(13)
def edit_key_camera(L,beta,d):
    n=f"{VL}.layers.{L}.self_attn.k_proj.weight"
    t=load_original(n)
    save_edit(n,t-beta*torch.outer(t@d,d))
edit_key_camera(14,0.5,cam14)
n=f"{VL}.layers.14.self_attn.k_proj.weight"
print("relative key change",float((load(n)-load_original(n)).norm()/load_original(n).norm()))
torch.save({"grip_input":g_specific,"grip_readout":g_read,"camera14":cam14},W/"analysis_directions.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Rejected candidate D, pass12\nGripper-directed O14 attenuation .5:26/48->26/48, "
            "2 rescues/2 regressions, loops21->20. Reset; gripper-only pathway edits not persuasive.\n"
            "\n## Candidate E: camera-offset key direction\nCompute wrist-minus-base axis from normalized "
            "VLM13 camera-group mean residuals (15 observations, independent of outcome labels), orthogonal "
            "to common visual and language means. Attenuate only this axis in VLM14 K by50%. "
            "Tests camera-source bias versus object content; also affects VLM14 self-attention, a caveat.\n")
```
