The residual identity reconstructs the recorded layer increments to within about 0.1–1.3% RMS, so the gate-aware attribution is usable. Late attention adds a much larger gripper-aligned residual in the sampled loops. I will check whether that difference survives command matching and build a source-specific direction that protects the other action readouts.
```python
print("L | attn grip effect(all/open) | MLP grip effect(all/open) | attn F/S open")
for l in [2,3,5,7,10,11,13,14,15,16,17]:
    av=grA[:,l-1].mean(1);mv=grM[:,l-1].mean(1)
    print(l,np.round([es(av,~succ19,succ19),es(av,~succ19&open19,succ19&open19),
                      es(mv,~succ19,succ19),es(mv,~succ19&open19,succ19&open19),
                      av[~succ19&open19].mean(),av[succ19&open19].mean()],3).tolist())
print("Action and final raw-grip residual check")
for i,t in enumerate(ident19):
    print(t[0],t[1],"S" if succ19[i] else "F",
          "grip",round(float(a19[i,:,6].mean()),2),
          "resid17",round(float(pR[i,17].mean()),2))
gate_directions={}
other=[i for i in range(32) if i!=6]
for l in [7,14,16,17]:
    G=ga[l-1].mean(axis=0)
    # Pull back each protected readout through the observed adaptive gate.
    protected=torch.from_numpy(G).float()[:,None]*Aout[other].T
    B=torch.linalg.qr(protected,mode="reduced").Q
    u=torch.from_numpy(G).float()*g
    u=u-B@(B.T@u)
    u=u/u.norm()
    gate_directions[l]=u
    proj=AT[:,l].mean(1)@u.numpy()
    print("gate-direction",l,
          "cos_raw_grip",round(float(u@g),4),
          "raw proj F/S",np.round([proj[~succ19].mean(),proj[succ19].mean()],3).tolist(),
          "effect open",round(float(es(proj,~succ19&open19,succ19&open19)),3),
          "protected leak",float((protected.T@u).norm()/protected.norm()))
torch.save(gate_directions,W/"gate_grip_directions.pt")
```
