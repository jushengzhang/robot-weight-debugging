The earlier physical-state probe also saved layer-15 attention outputs. Together with the residuals and measured adaptive gates, these let me reconstruct the pooled input to its MLP. I will use weight projections to identify channels that respond to the empty-hand contrast and preferentially write a closing command, while excluding channels with substantial pose coupling.
```python
# Reconstruct recorded, pooled MLP inputs algebraically; no network execution.
r27=json.loads((W/"probe_27"/"results.json").read_text())
a27={r["idx"]:r for r in r27 if r["arm"]=="orig"}
X14=torch.stack([torch.from_numpy(np.load(r["activations"])["calls::expert.14.resid::action"]).float() for r in rr26])
A15=torch.stack([torch.from_numpy(np.load(a27[i]["activations"])["calls::expert.15.attn_out::action"]).float() for i in range(len(phys))])
G15=torch.from_numpy(gm["L15_ga"]).float()
H15=X14+A15*G15
pm=C@load_original(E+"layers.15.post_attention_layernorm.dense.weight").T+load_original(E+"layers.15.post_attention_layernorm.dense.bias")
ps,ph,pg=pm.chunk(3,-1)
Y=H15/(H15.square().mean(-1,keepdim=True)+1e-6).sqrt()*(1+ps[None])+ph[None]
# A physical-state contrast; controls are separated by initial noise group.
protect=[Ain]
for gg in ["holding_like","open"]:
    for grip in ["open","close"]:
        ii=(groups==gg)&derive&np.array([p["grip"]==grip for p in phys])
        if ii.sum(): protect.append(Y[ii].mean(0).T)
PP=torch.cat(protect,1)
uu,sv,_=torch.linalg.svd(PP,full_matrices=False)
Pp=uu[:,sv>sv.max()*1e-5]
raw15=Y[(groups=="fully_closed")&derive].mean((0,1))-Y[(groups=="holding_like")&derive].mean((0,1))
d15=raw15-Pp@(Pp.T@raw15); d15/=d15.norm()
s15=(Y@d15).mean(1).numpy()
norm15=s15[(groups=="fully_closed")&derive].mean()
print("MLP15 cue check:",
      [(gg,round(float(s15[(groups==gg)&~derive].mean()/norm15),3)) for gg in ["fully_closed","holding_like","open"]])
gateW=load_original(E+"layers.15.mlp.gate_proj.weight")
upW=load_original(E+"layers.15.mlp.up_proj.weight")
downW=load_original(E+"layers.15.mlp.down_proj.weight")
readout=load_original("action_out_proj.weight")[:7]
fscale=1+tm["final_modulation"].float()[:,:1024]
# Negative Euler integration: a negative velocity contribution closes the gripper.
R=torch.einsum("od,td,td,dc->toc",readout,fscale,pg,downW)
pose_energy=R[:,:6].square().mean(0).sum(0)
grip_energy=R[:,6].square().mean(0)
gate_alignment=gateW@d15
empty_mean=Y[(groups=="fully_closed")&derive].mean(0)
gate_base=empty_mean@gateW.T
up_base=empty_mean@upW.T
closing_coupling=-(R[:,6]*up_base).mean(0)
eligible=(grip_energy>4*pose_energy)&(gate_alignment>0)&(closing_coupling>0)&(gate_base.mean(0)>0)
rank=gate_alignment*closing_coupling
rank[~eligible]=-torch.inf
idx15=rank.topk(8).indices
print("Eligible channels:",int(eligible.sum()))
print("Selected channels:",idx15.tolist())
print("Selected share of grip / pose coupling:",
      float(grip_energy[idx15].sum()/grip_energy.sum()),
      float(pose_energy[idx15].sum()/pose_energy.sum()))
print("Selected weight-sign evidence:",
      [(int(i),round(float(gate_alignment[i]),3),round(float(closing_coupling[i]),3),
        round(float(grip_energy[i]/pose_energy[i]),2)) for i in idx15])
torch.save({"indices":idx15,"cue_direction":d15,"derive_mask":torch.from_numpy(derive),
            "gate_alignment":gate_alignment,"closing_coupling":closing_coupling},
           W/"empty_closing_mlp15_channels.pt")
```
