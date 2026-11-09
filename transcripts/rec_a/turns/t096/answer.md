The reversed depth edit also failed overall (two rescues, six losses), so I will remove it. Before trying another component, I will check an important gap in the localization: the final adaptive normalization can rotate the effective action readout, so gate-adjusted weights alone may still misidentify the gripper-control heads. I can recover its time-dependent coefficients algebraically from the recorded residual gates, without running or training the network.
```python
for n in list(edited()): drop_edit(n)
(W/"current_candidate.json").write_text(json.dumps({"name":"none"}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 30: empty-cue depth correction rejected\n"
            "- Beta=-2 paired screen: 31/64 -> 27/64, -6.2 pp (SE 4.4), 2 rescues / 6 losses.\n"
            "- Loop failures 21 -> 24. Both signs of the cue-to-Z edit damage solved scenes. Reverted.\n"
            "- Forward budget 30/50; no current edits.\n")
As=[]; Ys=[]
for L in [14,15,16,17]:
    nm=f"{expert}layers.{L}.input_layernorm.dense"
    ww=load_original(nm+".weight").double()
    bb=load_original(nm+".bias").double()
    As.append(ww[2048:])
    Ys.append(torch.from_numpy(gates[f"L{L}_ga"]).double().T-bb[2048:,None])
A=torch.cat(As,0); Y=torch.cat(Ys,0)
# Recover an activation, not weights: solve the existing affine gate equations.
normal=A.T@A
cond=torch.linalg.solve(normal+1e-8*torch.eye(1024,dtype=torch.float64), A.T@Y).T.float()
print("Recovered time-conditioning vectors",list(cond.shape),"norm",np.round(cond.norm(dim=1).numpy(),2).tolist())
print("Equation residual RMS",float((A@cond.double().T-Y).square().mean().sqrt()))
for L in [10,14,15,16,17]:
    nm=f"{expert}layers.{L}.post_attention_layernorm.dense"
    ww=load_original(nm+".weight"); bb=load_original(nm+".bias")
    pred=cond@ww[2048:].T+bb[2048:]
    actual=torch.from_numpy(gates[f"L{L}_gm"]).float()
    print("Independent post-gate check",L,
          "relative error",round(float((pred-actual).norm()/actual.norm()),4),
          "cos",round(torch.nn.functional.cosine_similarity(pred.flatten(),actual.flatten(),dim=0).item(),5))
nm=f"{expert}norm.dense"
fw=load_original(nm+".weight"); fb=load_original(nm+".bias")
finalmod=cond@fw.T+fb
for j,lab in enumerate(["scale","shift","gate"]):
    q=finalmod[:,j*1024:(j+1)*1024]
    print("Final",lab,"mean/std",round(q.mean().item(),4),round(q.std().item(),4))
torch.save({"cond":cond,"final_modulation":finalmod},W/"recovered_time_modulation.pt")
```
