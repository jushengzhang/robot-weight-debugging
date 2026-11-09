I will check whether the probes expose per-token features or only pooled summaries, and measure where gripper flips occur relative to the five-action replanning boundary. This distinguishes a repeated visual decision from oscillation already built into each predicted chunk.

```python
pr8=json.loads((W/"probe_8/results.json").read_text())
print("residual result sample:",json.dumps(pr8[:1],ensure_ascii=True)[:2000])
p8act=pr8[0].get("activations")
if p8act:
    az=np.load(p8act)
    print("activation fields:",[(k,az[k].shape) for k in az.files][:38])
print("gripper flip locations (boundary vs within chunk):")
for groupname,rr in [("loop",loop1),("successful",success1)]:
    counts=np.zeros(5,dtype=int)
    per=[]
    for r in rr:
        a=np.load(r["actions_npy"])
        inds=np.flatnonzero(np.diff(a[:,6]>0))+1
        for j in inds: counts[j%5]+=1
        per.append((len(inds),len(a)))
    print(groupname,counts.tolist(),"boundary fraction",round(counts[0]/max(counts.sum(),1),3),
          "total flips",int(counts.sum()))
print("adaptive norm weight blocks: mean row norm; bias RMS; bias max")
for l in [0,1,2,7,8,14,15,16,17]:
    fields=[]
    for kind in ["input_layernorm","post_attention_layernorm"]:
        ww=load_original(f"{E}.layers.{l}.{kind}.dense.weight").reshape(3,1024,1024)
        bb=load_original(f"{E}.layers.{l}.{kind}.dense.bias").reshape(3,1024)
        fields.append(kind+":"+ ";".join(f"{ww[i].norm(dim=1).mean():.3f},{bb[i].square().mean().sqrt():.3f},{bb[i].abs().max():.3f}" for i in range(3)))
    print(l," ".join(fields))
print("initial input state (analysis only; model does not use it):")
for id_,gr in [(562,"open"),(562,"close"),(979,"open"),(979,"close")]:
    inp=torch.load(W/"roll_1_frames"/f"s{id_}_{gr}_seed7_original"/"t0000_inputs.pt",map_location="cpu",weights_only=False)
    st=np.asarray(inp["element"]["observation/state"])
    print(id_,gr,"state shape",st.shape,"last two",np.round(st[-2:],4).tolist(),
          "first5 grip",",".join(f"{v:.2f}" for v in np.asarray(inp["action_chunk"])[:5,6]))
with open(W/"notes.md","a") as f:
    f.write("\n## Passes 7-8\nL15H2 output factor0.5 likewise causes only0.000-0.004 XYZ RMS changes on4 diagnostic queries; not retained as a demonstrated repair. Broadened diagnosis to representation/self-attention bottlenecks; original residual probe8 saved.\n")
```
