Full removal did not improve the result, so I will reject the gate-corrected attenuation too. I will now test the opposite direction of an earlier causal finding: weakening layer 2, head 0’s gripper-specific query pathway caused three regressions and no rescues. Strengthening that pathway may improve action consistency rather than suppressing useful gripper information.
```python
for n in list(edited()):drop_edit(n)
# Recover the earlier causal probe and summarize head2.0's missing-prefix mass
# (the saved maps contain prefix attention, so the complement is action self-attention).
m9=json.loads((W/"probe_9"/"results.json").read_text())
for idx in sorted(set(r["idx"] for r in m9)):
    rr={r["arm"]:r for r in m9 if r["idx"]==idx}
    if not all(k in rr for k in ["orig","edit"]):continue
    sums=[]
    for arm in ["orig","edit"]:
        rr0=rr[arm]
        if "2" in rr0.get("attention",{}):
            q=np.load(localpath(rr0["attention"]["2"]))[0]
            sums.append(np.round([q[:,:256].sum(1).mean(),q[:,256:512].sum(1).mean(),
                                  q[:,768:].sum(1).mean(),1-q.sum(1).mean()],3).tolist())
    print("prior Q2 weakening",idx,"base/wrist/lang/self orig/edit",sums)
gi=directions["grip_input"].float();gi=gi/gi.norm()
qn=f"{E}2.self_attn.q_proj.weight"
Q=load_original(qn);Qe=Q.clone()
alpha=.5
Qe[:256]+=alpha*torch.outer(Q[:256]@gi,gi)
save_edit(qn,Qe)
Ain=load_original([n for n in NAMES if n.endswith("action_in_proj.weight")][0])
print("Saved selective Q2h0 gripper amplification",alpha,
      "relative change",float((Qe-Q).norm()/Q.norm()))
print("Relative Q2h0 changes by physical input column",
      [round(float(((Qe[:256]-Q[:256])@Ain[:,i]).norm()/
                   (Q[:256]@Ain[:,i]).norm()),4) for i in range(7)])
with (W/"notes.md").open("a") as f:
    f.write("\n## Gate-aware G beta1.0 rejected, pass23\n"
            "26/48->26/48,2 rescues/2 regressions,loops21->20; no monotone dose-response. Reset.\n"
            "\n## Candidate H\n"
            "Reverse earlier harmful Q2h0 gripper-path attenuation: now Q2h0 +=.5*(Q2h0*g_input)*g_input^T. "
            "g_input orthogonal to other31 action input columns. Prior beta+.5 suppression caused0 rescues/3 regressions. "
            "Test amplification as action-consistency intervention, not global action scaling.\n")
```
