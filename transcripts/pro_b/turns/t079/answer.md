Full DC removal still left environment success unchanged, so I will revert both DC candidates. I will inspect the action expert’s last layer and action-input geometry next. The aim is to find a concrete source of closed-loop instability, rather than keep modifying visual features that have not produced a reliable gain.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 25: DC full-dose rejected\n"
            "84/128->85/128 (+0.8 +/-3.0pp), environment44/80 unchanged again. "
            "Loop failures lower but no success dose-response; both doses rejected, original restored.\n")
ain=load_original("action_in_proj.weight")
Uact=torch.linalg.qr(ain[:,:7],mode="reduced").Q
print("Action input column norms:")
print(" used",[round(float(x),3) for x in ain[:,:7].norm(dim=0)])
print(" padded min/mean/max",[round(float(x),3) for x in
                                 [ain[:,7:].norm(dim=0).min(),ain[:,7:].norm(dim=0).mean(),
                                  ain[:,7:].norm(dim=0).max()]])
print("Padded input weight energy overlapping used-action span:",
      round(float((Uact.T@ain[:,7:]).square().sum()/ain[:,7:].square().sum()),4))
print("Action output used/padded row norms:",
      [round(float(x),3) for x in load_original("action_out_proj.weight").norm(dim=1)])
# Expert last-layer magnitudes and dominant directions, from saved activations only.
mlp13=[dict(np.load(x["activations"])) for x in pr[13]]
for l in [14,15,16,17]:
    X=np.stack([a[f"calls::expert.{l}.mlp_out::action"] for a in mlp13[:30]])
    means=X.mean(0)
    centered=X-means[None,:,:]
    v=np.sqrt((X*X).mean((0,1)))
    idx=np.argsort(v)[-6:][::-1]
    print("L",l,"MLP RMS norm",round(float(np.linalg.norm(X,axis=-1).mean()),2),
          "scene-variable energy fraction",round(float((centered*centered).sum()/(X*X).sum()),3),
          "top channels",[(int(i),round(float(v[i]),2)) for i in idx])
print("L17 instruction-swap absolute/relative differences:")
for t in [0,4,9]:
    for l in [14,16,17]:
        A,B=swap_act[0]
        aa=A[f"calls::expert.{l}.resid::action"][t]
        bb=B[f"calls::expert.{l}.resid::action"][t]
        print(t,l,"norm",round(float(np.linalg.norm(aa)),1),
              "delta",round(float(np.linalg.norm(bb-aa)),1))
```
