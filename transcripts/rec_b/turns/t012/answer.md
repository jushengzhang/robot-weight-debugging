I will compare the additional attention maps and inspect the candidate heads’ query/output weights. This will determine whether a narrowly targeted head intervention is justified and whether it would directly disturb the gripper readout or primarily change visual evidence selection.
```python
p3=json.loads((W/"probe_3/results.json").read_text())
p4=json.loads((W/"probe_4/results.json").read_text())
for k,pp in [(3,p3),(4,p4)]:
    print("PROBE",k)
    for L in [3,5,7,11]:
        print(" layer",L)
        for i,p in enumerate(pp):
            m,gr,ent=attn_stats(np.load(p["attention"][str(L)]))
            # compact visual-attention and self-attention for all heads
            print(i,"B",np.rint(100*gr[:,0]).astype(int).tolist(),
                  "W",np.rint(100*gr[:,1]).astype(int).tolist(),
                  "self",np.rint(100*(1-gr.sum(1))).astype(int).tolist())
print("candidate head's dominant tokens:")
for k,pp in [(2,p2),(3,p3),(4,p4)]:
    for i,p in enumerate(pp):
        m,gr,ent=attn_stats(np.load(p["attention"]["7"]))
        top=np.argsort(m[7])[-5:][::-1]
        print(k,i,"head7",[(int(j),round(float(m[7,j]),3)) for j in top])
EX="paligemma_with_expert.gemma_expert.model"
VL="paligemma_with_expert.paligemma.model.language_model"
for L in [3,5,7,11,14]:
    q=load_original(f"{EX}.layers.{L}.self_attn.q_proj.weight").reshape(8,256,1024)
    o=load_original(f"{EX}.layers.{L}.self_attn.o_proj.weight").reshape(1024,8,256)
    print("weights",L,"qnorm",q.flatten(1).norm(dim=1).numpy().round(2).tolist(),
          "onorm",o.permute(1,0,2).flatten(1).norm(dim=1).numpy().round(2).tolist(),
          "readout grip projection",(torch.einsum("d,dhk->hk",out_w[6]/out_w[6].norm(),o).norm(dim=1)).numpy().round(3).tolist())
```
