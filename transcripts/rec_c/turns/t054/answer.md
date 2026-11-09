After the noise control, expert layer 7 is the clearest repeatable candidate: its contrast is similar at steps 50 and 250 (cosine 0.93) and agrees moderately across gripper starts (0.43). This is still only correlation. I will halve that one MLP output direction, after removing the common gripper/noise direction and protecting the successful-recovery mean, then test its causal effect.
```python
L=7
k=f"mean::expert.{L}.mlp_out::action"
xx=X[k]
# Nuisance direction: initial close-minus-open, independent of outcome.
initinds=[i for i,r in enumerate(mf) if r["phase"]=="initial"]
closeinds=[i for i in initinds if mf[i]["grip"]=="close"]
openinds=[i for i in initinds if mf[i]["grip"]=="open"]
nuisance=xx[closeinds].mean(0)-xx[openinds].mean(0)
v=tm[L]
basis=torch.tensor(np.stack([v["success_mean"],nuisance],axis=1),dtype=torch.float32)
Q,_=torch.linalg.qr(basis,mode="reduced")
du=torch.tensor(v["difference"],dtype=torch.float32)
u=du-Q@(Q.T@du);u/=u.norm()
print("direction retained",round(float(u@du/du.norm()),4),
      "mean preserved",float(u@basis[:,0]),"nuisance overlap",float(u@basis[:,1]))
name=f"paligemma_with_expert.gemma_expert.model.layers.{L}.mlp.down_proj.weight"
weight=load_original(name)
alpha=0.5
delta=alpha*torch.outer(u,u@weight)
save_edit(name,weight-delta)
print("relative weight change",round(float(delta.norm()/weight.norm()),5))
torch.save({"u":u,"layer":L,"nuisance":torch.tensor(nuisance),
            "success_mean":torch.tensor(v["success_mean"]),"alpha":alpha},
           W/"loop_mlp_direction_L7.pt")
print("backup",backup_edits("candidate_L7_mlp_loop_half"))
probeinputs=[mf[i]["input"] for i in range(2,len(mf),4)]+[r["input"] for r in mf20]
(W/"mlp_causal_probe_inputs.json").write_text(json.dumps(probeinputs))
print("probe inputs",probeinputs)
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes19-20: MLP diagnosis with noise controls\n"
            "Matched16 scenes: one success/one loop,8 each successful start grip. "
            "Initial and step50 probes, plus failure step250. Comparing different policy times "
            "creates a severe saved-noise confound in early expert layers; corrected by matching step50. "
            "L7 MLP contrast remains similar at50/250 (cos0.928), split by successful start grip cos0.430. "
            "Candidate: rank-one halve L7 MLP output along time-matched loop-minus-success direction, "
            "orthogonalized against successful centroid and initial close-open nuisance direction. "
            "No other tensors changed; needs causal and independent evaluation.\n")
del weight,delta
```
