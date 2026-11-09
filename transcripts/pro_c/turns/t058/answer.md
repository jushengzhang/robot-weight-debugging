I will compare wrist, base, BOS, and instruction-content attention separately. BOS attention is not instruction following, so increases in the whole language segment alone would be misleading.
```python
ix19=json.loads((W/"probe_19"/"results.json").read_text())
print("arms",Counter(r["arm"] for r in ix19))
pairs19=defaultdict(dict)
for r in ix19: pairs19[r["idx"]][r["arm"]]=r
for l in [4,8,12,16]:
    agg=defaultdict(list)
    for idx,pair in pairs19.items():
        for arm,r in pair.items():
            x=np.load(r["attention"][str(l)]).mean(1)
            nv=r["prefix_layout"]["n_lang_valid"]
            agg[arm].append(np.stack([x[:,:256].sum(1),x[:,256:512].sum(1),x[:,768],x[:,769:768+nv].sum(1)],-1))
    oo=np.mean(agg["orig"],0); ee=np.mean(agg["edit"],0)
    print("layer",l,"[base,wrist,BOS,words] percentages")
    for h in range(8):
        print(h,np.round(100*oo[h],1).tolist(),"->",np.round(100*ee[h],1).tolist())
for i,p in pairs19.items():
    a=np.load(p["orig"]["action"]); b=np.load(p["edit"]["action"])
    print("action",i,"RMS delta",np.round(np.sqrt(np.mean((b-a)**2,0)),4).tolist())
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate G / pass19\nRank1 wrist-common pre-LN key component removed at50% in VLM8/12, preserving span of32 base means,32 lang means, sink coords. ~1.6-1.8% relative weight changes. Exact paired attention probe4 inputs completed; routing measured separately for BOS/content.\n")
```
