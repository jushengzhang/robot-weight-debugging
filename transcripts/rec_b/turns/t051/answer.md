The residual analysis does not yet isolate a safe MLP repair. I will instead finish testing the camera-key intervention already probed: it reduced wrist dominance and changed loop-state motion more than successful-control motion. The edit removes half of one camera-identity direction from VLM layer 14’s key projection, leaving orthogonal directions unchanged.
```python
for n in list(edited()): drop_edit(n)
V="paligemma_with_expert.paligemma.model.language_model.layers."
kn=f"{V}14.self_attn.k_proj.weight"
c=directions["camera14"].float();c=c/c.norm()
K=load_original(kn)
beta=.5
Ke=K-beta*torch.outer(K@c,c)
save_edit(kn,Ke)
print("Saved",kn,"rank1 beta",beta,"relative change",float((Ke-K).norm()/K.norm()))
# Check that the saved camera axis generalizes to the larger observational sample.
deltas=[]
for z in acts19:
    b=z["mean::vlm.13.resid::base"];w=z["mean::vlm.13.resid::wrist"]
    b=b/(np.linalg.norm(b)+1e-9);w=w/(np.linalg.norm(w)+1e-9)
    deltas.append(w-b)
deltas=np.stack(deltas)
print("Axis separation F/S",
      float((deltas@c.numpy())[~succ19].mean()),float((deltas@c.numpy())[succ19].mean()))
print("Per-observation sign consistency",int(((deltas@c.numpy())>0).sum()),"/",len(deltas))
with (W/"notes.md").open("a") as f:
    f.write("\n## Expanded diagnosis, passes18-19\n"
            "44 MLP probes (11 task-matched success/failure grip pairs,start+late),22 late residual probes. "
            "Raw gripper differences are partly confounded by open/closed command; no MLP edit justified yet. "
            "Residual differences grow around L7, but adaptive gates complicate raw-readout attribution. "
            "Next complete rollout test of previously probed camera-key edit:K14 rank1 camera-axis attenuation beta=.5.\n")
```
