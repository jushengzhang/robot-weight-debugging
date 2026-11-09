Halving head 14 did not help: the paired change was -0.8 percentage points with a 2.6-point standard error. I will revert it. Next I will analyze the shared directions of the layer-11 attention-output contrasts and identify which projection matrices can write those directions, rather than treating a large weight norm as evidence of a defect.
```python
clear_edits()
r7=[json.loads(s) for s in (W/"roll_7.jsonl").read_text().splitlines()]
print("Reverted; active edits:",edited())
print("\nLayerwise contrast concentration:")
pcs={}
for l in [0,9,10,11,12,13,15,17,27]:
    d=torch.from_numpy(AD[:,l]).float()
    dn=d/torch.linalg.vector_norm(d,dim=1,keepdim=True).clamp_min(1e-8)
    _,s,vh=torch.linalg.svd(dn,full_matrices=False)
    u=vh[0]
    pcs[l]=u
    coord=d.square().sum(0)
    ix=torch.topk(coord,4).indices.tolist()
    print(l,"PC1 pair-normalized frac",round(float(s[0]**2/s.square().sum()),3),
          "raw top coords", [(i,round(float(coord[i]/coord.sum()),3)) for i in ix],
          "u largest",[(int(i),round(float(u[i]),3)) for i in torch.topk(u.abs(),3).indices])
print("\nL11 shared contrast projections into each pair:")
u=pcs[11]
for i in range(10):
    g,b=torch.from_numpy(A[2*i,11]),torch.from_numpy(A[2*i+1,11])
    d=b-g
    print(i,"g",round(float(g@u),3),"b",round(float(b@u),3),
          "direction cosine",round(float(d@u/torch.linalg.vector_norm(d)),3))
print("\nL11 output reader/writer alignment:")
for e in ["action"]: 
    for att in ["self_attn","cross_attn"]:
        ow=load_original(f"mot.mixtures.{e}.blocks.11.{att}.o.weight")
        proj=u@ow
        he=proj.reshape(24,128).square().sum(1)
        print(att, "output projection norm",round(float(proj.norm()),3),
              "top heads",[(int(h),round(float(he[h]/he.sum()),3))
                           for h in torch.topk(he,6).indices])
torch.save(pcs,W/"paired_attention_contrast_PCs.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 7 causal negative result\n"
            "- L13 H14 output gain 0.5 in both experts, paired same 128 discovery variants: original 90/128, edited 89/128; -0.8 pp +/-2.6 pp SE; 5 rescues, 6 regressions. Head gain hypothesis not supported. REVERTED all edits.\n"
            "- Next localization: paired instruction-contrast directions at L11 attention outputs; seek a low-rank, task-preserving intervention and test its action prediction effect before rollout.\n")
```
