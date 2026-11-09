The traces do not support a simple “closing while moving too fast” explanation: successful grasps actually have larger approach commands on average. I will not use that as a repair premise. Instead, I will examine the small set of late MLP channels that account for much of the gated gripper coupling, and test whether they form a selective gripper-feedback pathway.
```python
mlp_candidates={}
print("MLP specialization: layer, top32 grip-energy share, pose-energy share, median input-contrast alignment percentile")
for l in (14,15,16,17):
    gm=torch.from_numpy(branch_data[l]["gm"].astype(np.float32))
    dw=load_original(E+f"layers.{l}.mlp.down_proj.weight")
    # Effective readout over the first half of the denoising trajectory.
    gc=(gm[:5]*outw[6])@dw
    pc=torch.einsum("td,ad,dn->tan",gm[:5],outw[:6],dw)
    ge=gc.square().mean(0)
    pe=pc.square().mean((0,1))
    # Large actual gripper coupling, with a minimum 4:1 squared grip/pose preference.
    eligible=ge>4*pe
    score=ge.clone()
    score[~eligible]=-1
    ix=score.topk(32).indices

    Rprev=np.stack([a[f"calls::expert.{l-1}.resid::action"][0] for a in a4])
    P=Rprev+branch_data[l]["A"][:,0]*branch_data[l]["ga"][0]
    close=grip4>0.5; op=grip4<-.5
    d=torch.tensor(P[close].mean(0)-P[op].mean(0),dtype=torch.float32)
    d/=d.norm()
    up=load_original(E+f"layers.{l}.mlp.up_proj.weight")
    ga=load_original(E+f"layers.{l}.mlp.gate_proj.weight")
    align=(up@d).abs()/up.norm(dim=1)
    ranks=torch.argsort(torch.argsort(align)).float()/len(align)
    mlp_candidates[l]={"indices":ix,"grip_energy":ge,"pose_energy":pe,
                       "input_contrast":d,"up_alignment":align}
    print(l,round(float(ge[ix].sum()/ge.sum()),3),
          round(float(pe[ix].sum()/pe.sum()),3),
          round(float(ranks[ix].median()),3),
          "max grip/pose ratio",round(float((ge/pe.clamp_min(1e-20)).max()),1))
torch.save(mlp_candidates,W/"gripper_mlp_channels.pt")

# Single-layer, channel-selective causal intervention.
for n in list(edited()):
    drop_edit(n)
layer=16; gain=0.5
ix=mlp_candidates[layer]["indices"]
name=E+f"layers.{layer}.mlp.down_proj.weight"
w=load_original(name)
w[:,ix]*=gain
save_edit(name,w)
spec={"name":"E16_MLP_grip32_gain0.5","layer":16,"gain":gain,
      "indices":ix.tolist(),"selection":"top gated gripper-coupling channels, grip/pose energy > 4"}
(W/"current_candidate.json").write_text(json.dumps(spec,indent=2))
print("Active candidate",spec["name"],"channels",ix.tolist())
with (W/"notes.md").open("a") as f:
    f.write("\n## Transition panel and sparse MLP feedback hypothesis\n"
            "- Saved 64 transition-focused original inputs, 16 each loop/success x closing/opening, in transition_probe_manifest.json.\n"
            "- Executed traces do NOT show higher approach speed at closing in loops: pre-close XYZ norm 0.333 loops vs 0.419 successes. No premature-speed claim.\n"
            "- Candidate: halve 32 E16 MLP down-projection columns selected by gated gripper coupling and >4:1 squared grip/pose preference. No full-tensor scaling.\n")
```
