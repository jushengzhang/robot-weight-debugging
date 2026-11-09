The recorded branches reconstruct the late-layer increments to within 0.5% error. Both attention and MLP updates contribute substantially to the gripper signal. Crucially, the adaptive multipliers change which output directions matter, so I will rank the late heads using those measured multipliers rather than the raw readout alignment used earlier.
```python
weighted_head_scores=[]
print("Late-layer head coupling after the measured adaptive residual gate:")
for l in (14,15,16,17):
    D=torch.from_numpy(branch_data[l]["ga"].astype(np.float32))
    wg=outw[6]*D[0]
    wp=outw[:6]*D[0]
    wo=load_original(E+f"layers.{l}.self_attn.o_proj.weight")
    print("Layer",l,"gate-readout cos first/last",round(float(torch.nn.functional.cosine_similarity(wg,outw[6]*D[-1],dim=0)),3))
    for h in range(8):
        oh=wo[:,h*256:(h+1)*256]
        gc=float((wg@oh).norm())
        pc=float((wp@oh).norm()/math.sqrt(6))
        mass=[x[5] for x in couplings if x[0]==l and x[1]==h][0]
        weighted_head_scores.append((l,h,gc,pc,gc/pc,mass[3]))
        print(h,"g",round(gc,3),"pose",round(pc,3),"ratio",round(gc/pc,2),
              "self",round(float(mass[3]),3))
print("Strongest gated grip-to-pose heads:")
for row in sorted(weighted_head_scores,key=lambda x:-x[4])[:8]:
    print(tuple(round(float(x),3) if isinstance(x,(float,np.floating)) else x for x in row))
print("Late MLP grip coupling concentration after adaptive gate:")
for l in (14,15,16,17):
    D=torch.from_numpy(branch_data[l]["gm"].astype(np.float32))
    dw=load_original(E+f"layers.{l}.mlp.down_proj.weight")
    z=(outw[6]*D[0])@dw
    energy=z.square()
    print(l,"gated grip norm",round(float(z.norm()),3),
          "top 8/32 shares",round(float(energy.topk(8).values.sum()/energy.sum()),3),
          round(float(energy.topk(32).values.sum()/energy.sum()),3),
          "gate-readout t0/t9 cosine",round(float(torch.nn.functional.cosine_similarity(outw[6]*D[0],outw[6]*D[-1],dim=0)),3))
```
