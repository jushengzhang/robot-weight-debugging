The source attenuation moved channel 1938 only slightly and did not support a repair, so I will leave that route unchanged. The more direct observation is the large instruction-induced redistribution of visual attention at layer 11. I will quantify which heads become more concentrated on failed prompts and test a head-local change only if the maps support it.
```python
head_map_stats=[]
for l in [10,11,13,15]:
    mm=[np.load(r["attention"][str(l)]) for r in pr5]
    for h in range(24):
        vals=[]
        for j in range(2):
            aa,bb=mm[2*j][h],mm[2*j+1][h]
            # Normalize within visual tokens; retain action-vs-video mass separately.
            ap=aa/(aa.sum(-1,keepdims=True)+1e-12)
            bp=bb/(bb.sum(-1,keepdims=True)+1e-12)
            ent=lambda x:-(x*np.log(x+1e-12)).sum(-1).mean()
            vals.append({"l1":float(np.abs(ap-bp).sum(-1).mean()),
                         "entropy_good":float(ent(ap)),"entropy_bad":float(ent(bp)),
                         "max_good":float(ap.max(-1).mean()),"max_bad":float(bp.max(-1).mean())})
        head_map_stats.append({"layer":l,"head":h,
                              "mean_l1":np.mean([x["l1"] for x in vals]),
                              "entropy_change":np.mean([x["entropy_bad"]-x["entropy_good"] for x in vals]),
                              "pairs":vals})
print("Top visual redistribution heads at L11:")
for x in sorted([x for x in head_map_stats if x["layer"]==11],
                key=lambda x:x["mean_l1"],reverse=True)[:8]:
    print(compact_obj(x,900))
print("\nMost consistent concentration increases across measured layers:")
xs=[x for x in head_map_stats if all(p["entropy_bad"]<p["entropy_good"] for p in x["pairs"])]
for x in sorted(xs,key=lambda x:x["entropy_change"])[:8]:
    print("L",x["layer"],"H",x["head"],"L1",round(x["mean_l1"],3),
          "entropy change",round(x["entropy_change"],3),
          "max pairs",[(round(p["max_good"],3),round(p["max_bad"],3)) for p in x["pairs"]])
print("\nL11 self-attention O head Frobenius / median:")
O=load_original("mot.mixtures.action.blocks.11.self_attn.o.weight")
hn=O.reshape(1024,24,128).square().sum((0,2)).sqrt()
print([(h,round(float(hn[h]/hn.median()),3)) for h in [2,4,5,15,19,21]])
(W/"visual_attention_head_diagnosis.json").write_text(json.dumps(head_map_stats,indent=2))
```
