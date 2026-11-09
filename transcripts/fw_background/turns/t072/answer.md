After protecting light-task variation, one leading video-residual direction remains strongly texture-sensitive across depth; the second direction is less reproducible, so I will leave it untouched. I will attenuate only the leading direction in each video value matrix, with no action, key, query, bias, or scaling changes. The fixed-input probe will test whether this trajectory-derived estimate transfers to identical initial states.
```python
# Compare the leading nuisance direction with a basis using only pairs where both rollouts succeeded.
both_success=[p for p in validpairs if rlookup[(p["suite"],p["high"])]["success"]
              and rlookup[(p["suite"],p["low"])]["success"]]
print("Both-success pairs for confound check:",len(both_success))
for L in [2,10,15,20,25,29]:
    keys=[f"mean::video.{L-1}.resid::{cam}" for cam in ["agent","wrist"]]
    B=standardized_summary(np.stack([[vr[(p["suite"],p[z])][k]
        for z in ["high","low"] for k in keys] for p in both_success]))
    D=(B[:,:2]-B[:,2:]).reshape(-1,3072)
    Q=vres_data[f"Q{L}"]; Dp=D-(D@Q)@Q.T
    Vs=np.linalg.svd(Dp,full_matrices=False)[2][:2].T
    u=vres_data[f"U{L}"][:,0]
    print("L",L,"leading direction overlap with success-only rank2:",
          round(float(np.linalg.norm(Vs.T@u)),3))

def apply_video_values(dose=0.5, control=False, layers=tuple(range(1,30))):
    clear_edits()
    rep=[]; coupling={}
    rng=np.random.default_rng(472937)
    for L in layers:
        name=f"mot.mixtures.video.blocks.{L}.self_attn.v.weight"
        wo=load_original(name)
        U=torch.from_numpy(vres_data[f"U{L}"][:,:1].copy()).float()
        target=-dose*(wo@U)@U.T
        eff=dose
        if control:
            Q=torch.from_numpy(vres_data[f"Q{L}"].copy()).float()
            R=torch.from_numpy(rng.standard_normal((3072,1)).astype(np.float32))
            R=R-Q@(Q.T@R)-U@(U.T@R)
            U=R/R.norm()
            raw=(wo@U)@U.T
            eff=float(target.norm()/raw.norm())
        delta=-eff*(wo@U)@U.T
        save_edit(name,wo+delta)
        rep.append({"layer":L,"rank":1,"effective_dose":eff,
                    "weight_relative_change":float(delta.norm()/wo.norm()),
                    "normal_mean_leak":float((U.T@torch.from_numpy(vres_data[f"mean{L}"])).norm())})
        if not control and L in [2,10,14,20,25,29]:
            av=load_original(f"mot.mixtures.action.blocks.{L}.self_attn.o.weight")
            vu=(wo@U).reshape(24,128)
            scores=torch.stack([(av[:,h*128:(h+1)*128]@vu[h]).norm() for h in range(24)])
            coupling[L]=scores.numpy()
    spec={"family":"protected_video_value_input_projection","layers":list(layers),
          "rank":1,"dose":dose,"control":control,
          "protected":"light means, camera mean difference, constant, top8 light PCs",
          "tensors":rep}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
    return spec,coupling

vcandidate,vcoupling=apply_video_values(0.5)
print("Video value edit matrix relative change range:",
      min(x["weight_relative_change"] for x in vcandidate["tensors"]),
      max(x["weight_relative_change"] for x in vcandidate["tensors"]))
print("Max light-summary mean projection:",max(x["normal_mean_leak"] for x in vcandidate["tensors"]))
for L,scores in vcoupling.items():
    print("Dominant action heads reading nuisance, L",L,
          [(int(h),round(float(scores[h]),3)) for h in np.argsort(-scores)[:5]])
np.savez(W/"video_nuisance_action_head_coupling.npz",
         **{f"L{L}":v for L,v in vcoupling.items()})
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 14: observation-side protected nuisance direction\n"
            "- Original video residuals recorded on 102 diagnostics. These are episode averages, so trajectory confounding remains; fixed-input causal probes must verify transfer.\n"
            "- Standardized camera summaries; protect both light means, constant direction and top8 light-task PCs. Leading residual contrast is highly texture-specific, with contrast/light variance ratios ~100-1500 and <1% light variance. Second direction is unstable and excluded.\n"
            "- Candidate: V_l' = V_l - .5 (V_l u_l) u_l^T, video value weights only, layers1-29, rank1 each; u_l from prior residual block. Action/Q/K/biases unchanged.\n"
            "- Passes used14/40. Experimental, not accepted.\n")
```