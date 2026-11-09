I will rank late heads by their visual versus self-attention patterns and compare their output coupling to translation and gripper directions. The next causal edit will be confined to one localized pathway, with the unsuccessful alternatives retained in the notes.
```python
p11=json.loads((W/"probe_11/results.json").read_text())
for L in [12,15,16,17]:
    print("layer",L)
    for i,p in enumerate(p11):
        m,gr,ent=attn_stats(np.load(p["attention"][str(L)]))
        print(i,"B",np.rint(100*gr[:,0]).astype(int).tolist(),
              "W",np.rint(100*gr[:,1]).astype(int).tolist(),
              "self",np.rint(100*(1-gr.sum(1))).astype(int).tolist(),
              "top",np.argmax(m,axis=1).tolist())
# Use episode summaries to inspect residual attention-output changes along each readout direction.
for L in [11,12,13,14,15,16,17]:
    key=f"mean::expert.{L}.attn_out::action"
    xs=np.stack([A[r["id"],r["grip"]][key] for r in valid])
    loops=np.array([not r["success"] and r["grip_flips"]>=6 for r in valid])
    succ=np.array([r["success"] for r in valid])
    proj=xs@out_w[:7].numpy().T
    effect=(proj[loops].mean(0)-proj[succ].mean(0))/np.sqrt((proj[loops].var(0)+proj[succ].var(0))/2).clip(1e-5)
    print("L",L,"readout effects xyz/rpy/grip",effect.round(2).tolist())
    o=load_original(f"{EX}.layers.{L}.self_attn.o_proj.weight").reshape(1024,8,256)
    read=torch.einsum("ad,dhk->ahk",out_w[:7]/out_w[:7].norm(dim=1,keepdim=True),o).norm(dim=2)
    print(" grip/xyz coupling",(read[6]/read[:3].mean(0)).numpy().round(2).tolist())
```
