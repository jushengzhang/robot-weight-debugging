The saved maps show several late heads attending almost entirely to action tokens in both successful and looping observations. I will check which of these heads couples most strongly to the gripper readout before choosing a single-head intervention. Self-attention by itself is not evidence of a fault.
```python
couplings=[]
for l in range(18):
    wo=load_original(E+f"layers.{l}.self_attn.o_proj.weight")
    for h in range(8):
        oh=wo[:,h*256:(h+1)*256]
        grip=float((outw[6]@oh).norm())
        pose=float((outw[:6]@oh).norm()/math.sqrt(6))
        masses=[m[h] for (k,i,ll),m in head_attn.items() if ll==l]
        mass=np.mean(masses,axis=0) if masses else np.full(4,np.nan)
        couplings.append((l,h,grip,pose,grip/(pose+1e-12),mass))
print("Top grip-to-pose output coupling heads:")
for l,h,gg,pp,rr,mm in sorted(couplings,key=lambda z:-z[4])[:18]:
    print(l,h,round(gg,3),round(pp,3),round(rr,3),"B/W/lang/self",np.round(mm,2).tolist())
print("Self-only / strongly-self heads:")
for l,h,gg,pp,rr,mm in couplings:
    if mm[3]>0.70:
        print(l,h,round(gg,3),round(pp,3),round(rr,3),"B/W/lang/self",np.round(mm,3).tolist())
print("Late residual norm parameter chunks:")
for layer in (14,15,16,17):
    bn=load_original(E+f"layers.{layer}.input_layernorm.dense.bias")
    wn=load_original(E+f"layers.{layer}.input_layernorm.dense.weight")
    print(layer,"bias RMS", [round(float(x.square().mean().sqrt()),3) for x in bn.chunk(3)],
          "weight RMS",[round(float(x.square().mean().sqrt()),4) for x in wn.chunk(3)])
bn=load_original(E+"norm.dense.bias"); wn=load_original(E+"norm.dense.weight")
print("Final norm bias RMS", [round(float(x.square().mean().sqrt()),3) for x in bn.chunk(3)],
      "weight RMS",[round(float(x.square().mean().sqrt()),4) for x in wn.chunk(3)])
print("Late raw residual grip signal at first/last denoising call, across p15 inputs:")
for l in (0,3,6,9,12,14,15,16,17):
    R=np.stack([a[f"calls::expert.{l}.resid::action"] for a in a15])
    gp=R @ g.numpy()
    grip=ac15[:,:,6].mean(1)
    print(l,"RMS",round(float(np.sqrt(np.mean(R**2))),2),
          "grip-axis std",np.round(gp.std(0)[[0,9]],2).tolist(),
          "corr",round(float(np.corrcoef(gp[:,0],grip)[0,1]),3),
          round(float(np.corrcoef(gp[:,-1],grip)[0,1]),3))
```
