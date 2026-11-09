I will combine the new maps with a weight-space check of each head's access to the gripper readout. This is a prioritization step, not a causal conclusion: an attention head with strong gripper coupling is only a repair candidate if selectively changing it produces the predicted behavioral effect.
```python
p3=json.loads((W/"probe_3/results.json").read_text())
p4=json.loads((W/"probe_4/results.json").read_text())
print("Residual probe record keys:",p4[0].keys())
print("Residual probe record summary:",str(p4[0])[:1100])
p4acts=np.load(p4[0]["activations"])
print("Residual shapes:",[(k,p4acts[k].shape) for k in p4acts.files])
att3={}
for p in p3:
    ii=p["idx"]; att3[ii]={}
    for l,fp in p["attention"].items():
        att3[ii][int(l)]=np.load(fp).astype(np.float32).mean(1)
print("Attention summaries: layer/head, avg B/W/lang/self on loops; on successes; loop-self change")
for l in [0,6,10,17]:
    AL=(att3[0][l]+att3[2][l])/2
    AS=(att3[1][l]+att3[3][l])/2
    for h in range(8):
        def masses(A):
            return [A[h,:256].sum(),A[h,256:512].sum(),A[h,768:].sum(),1-A[h].sum()]
        ml,ms=masses(AL),masses(AS)
        print(l,h,"/".join(f"{v:.2f}" for v in ml),"/".join(f"{v:.2f}" for v in ms))
EX="paligemma_with_expert.gemma_expert.model"
VL="paligemma_with_expert.paligemma.model.language_model"
head_stats=[]
for l in range(18):
    O=load_original(f"{EX}.layers.{l}.self_attn.o_proj.weight").reshape(1024,8,256)
    gj=torch.einsum("i,ihd->hd",gdir,O).norm(dim=1)
    xyz=torch.einsum("oi,ihd->ohd",AW[:6]/AW[:6].norm(dim=1,keepdim=True),O).square().sum(dim=(0,2)).sqrt()/math.sqrt(6)
    norms=O.square().sum(dim=(0,2)).sqrt()
    for h in range(8):
        head_stats.append((l,h,float(gj[h]),float(gj[h]/xyz[h]),float(norms[h])))
print("Top 24 heads by raw gripper-readout coupling: L/H |g^T O_h| /mean_xyz norm")
for row in sorted(head_stats,key=lambda r:-r[2])[:24]:
    print(row[0],row[1],*(round(v,4) for v in row[2:]))
print("Top 12 heads by gripper/pose coupling ratio:")
for row in sorted(head_stats,key=lambda r:-r[3])[:12]:
    print(row[0],row[1],*(round(v,4) for v in row[2:]))
print("Adaptive norm bias block statistics (first/second/third 1024):")
for l in [0,3,8,12,15,17]:
    z=load_original(f"{EX}.layers.{l}.input_layernorm.dense.bias").reshape(3,1024)
    print(l,"means",ascii_round(z.mean(1)),"std",ascii_round(z.std(1)))
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 2-4: initial probes\n"
            "- Two matched task pairs: 367 (cream cheese/butter) and 753 (alphabet soup).\n"
            "- BOS and wrist attention are common in both successes and loops; not enough for an ablation claim.\n"
            "- Some mid-layer heads lose base-camera mass in looping observations; self-only heads also present.\n"
            "- Collected all expert/VLM residual layers for 16 observations in four matched pairs.\n"
            "- Important: final VLM block output is downstream of the last KV-cache creation; "
            "its large episode-level contrast is not itself a conditioning route.\n")
```
