I will measure the consistency of the open/closed visual contrast and, if it is coherent, attenuate that single feature by 50% in the multimodal projector. I will compensate the projector bias so the learned SigLIP layer-normalization offset is unchanged; this targets image-dependent aperture information rather than shifting all visual tokens uniformly.

```python
pr15=json.loads((W/"probe_15/results.json").read_text())
bp15={p["idx"]:{k:np.load(p["activations"])[k].astype(np.float32)
                 for k in np.load(p["activations"]).files if k.startswith("mean::")} for p in pr15}
Ds={}
for k in bp15[0]:
    d=np.stack([bp15[2*i][k]-bp15[2*i+1][k] for i in range(16)])
    Ds[k]=d
    nd=d/(np.linalg.norm(d,axis=1,keepdims=True)+1e-9)
    mu=nd.mean(0)
    u=mu/(np.linalg.norm(mu)+1e-9)
    sims=nd@u
    if "vision.26" in k or "vlm.2" in k or "vlm.15" in k:
        print(k,"mean contrast norm",f"{np.linalg.norm(d,axis=1).mean():.3f}",
              "mean-direction cos mean/min",f"{sims.mean():.3f}/{sims.min():.3f}",
              "mean-vector coherence",f"{np.linalg.norm(mu):.3f}")
dw=Ds["mean::vision.26.resid::wrist"]
dw_unit=dw/(np.linalg.norm(dw,axis=1,keepdims=True)+1e-9)
raw_grip_visual=torch.from_numpy(dw_unit.mean(0))
raw_grip_visual=raw_grip_visual-raw_grip_visual.mean()
visual_grip_direction=raw_grip_visual*lnw
visual_grip_direction=visual_grip_direction/visual_grip_direction.norm()
torch.save({"direction":visual_grip_direction,"raw_mean_direction":raw_grip_visual,
            "scene_ids":grip_feature_ids},W/"visual_grip_direction.pt")
# Split-half consistency is a diagnostic of the controlled contrast, not a performance fit.
d1=dw_unit[::2].mean(0);d2=dw_unit[1::2].mean(0)
print("wrist direction split-half cosine",f"{cosine(d1,d2):.3f}")
P="paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
def set_visual_grip_attenuation(alpha):
    clear_edits()
    m=load_original(P+".weight"); b=load_original(P+".bias")
    u=visual_grip_direction
    v=m@u
    delta=-float(alpha)*torch.outer(v,u)
    save_edit(P+".weight",m+delta)
    save_edit(P+".bias",b+float(alpha)*v*torch.dot(u,lnb))
    print("visual aperture attenuation",alpha,"relative W change",f"{float(delta.norm()/m.norm()):.5f}",
          "bias correction norm",f"{float((float(alpha)*v*torch.dot(u,lnb)).norm()):.5f}")
    print("retained orthogonal directions exactly in fp32; rank-one singular value",f"{float(alpha*v.norm()):.5f}")
set_visual_grip_attenuation(0.5)
checkpoint_edits("candidate_visual_grip_half","Projector rank-one attenuation of controlled wrist aperture contrast, alpha0.5, with LN-offset-preserving bias compensation.")
print("pre-fail proxy ids:",sorted({r["id"] for r in allbase if r["stage"]=="pre-fail"}))
with open(W/"notes.md","a") as f:
    f.write("\n## Pass15: controlled visual aperture direction\nProbed16 initial open/closed pairs across4 suites; only vision/VLM features used (noise-independent), no success labels. Direction = mean unit wrist contrasts at SigLIP layer26, centered and expressed with post-LN weight scaling. Saved visual_grip_direction.pt. Candidate: projector W'=W-alpha*(W u)u^T, b'=b+alpha*(W u)*(u dot postLN_bias), alpha0.5. This preserves the LN offset and all directions orthogonal to u. Downstream redundant gripper cues are targeted at their shared visual source. Need causal and independent evaluation.\n")
```
