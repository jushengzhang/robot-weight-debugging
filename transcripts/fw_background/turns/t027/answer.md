The largest Q/K gain coordinates occur late in the network, but they are not bias-dominated enough to establish an attention-sink defect. I will finish the weight reference by measuring each late head's direct residual-path coupling to the seven action outputs. This can help avoid suppressing a head that is especially important for gripper or orientation control.

```python
readout=load_original("mot.mixtures.action.head.weight")
late_action_geometry={}
for l in [24,25,26,27,28,29]:
    o=load_original(pname("action",l,"self_attn.o.weight"))
    v=load_original(pname("video",l,"self_attn.v.weight")).reshape(24,128,3072)
    ro=(readout@o).reshape(7,24,128).permute(1,0,2).contiguous()
    rv=torch.bmm(ro,v)
    energy=rv.square().sum(-1)
    late_action_geometry[l]=energy.numpy()
    pos=energy[:,:3].sum(1)
    ori=energy[:,3:6].sum(1)
    grip=energy[:,6]
    print("Layer",l,"direct-path strongest heads:",
          {"translation":torch.argsort(pos,descending=True)[:4].tolist(),
           "orientation":torch.argsort(ori,descending=True)[:4].tolist(),
           "gripper":torch.argsort(grip,descending=True)[:4].tolist()})
np.savez(W/"late_head_direct_readout_geometry.npz",
         **{f"L{l}":x for l,x in late_action_geometry.items()})
print("Final readout row norms:",np.round(readout.norm(dim=1).numpy(),4).tolist())
print("Original checkpoint still intact:",len(edited())==0)
note("- Late-head direct residual-path readout geometry saved (not a full-network influence estimate). "
     "Weight-only analysis is complete; no background-specific or causal conclusion yet.")
del readout,o,v,ro,rv
```
