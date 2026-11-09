The compact payload loaded successfully. I will now check that the untouched controls are still bit-identical, measure the new anchors' effects, and prepare the corresponding full-strength replay to complete the dose test.
```python
d11=probe_dict(11)
refmap={0:3,1:3,2:3,4:5,6:8,7:8,9:10}
for i,r in enumerate(probe_safe_records):
    ao=np.load(d11[i,"orig"]["action"]); ae=np.load(d11[i,"edit"]["action"])
    ids=set(sp.encode(r["instruction"]+"\n",add_bos=True))
    affected=ids & (set(pairs_small)|set(pairs_extra))
    out={"i":i,"id":r["id"],"changed_tokens":[piece_ascii(s) for s in affected],
         "delta_rms":round(float(np.sqrt(np.mean((ae-ao)**2))),4),"exact":bool(np.array_equal(ao,ae))}
    if i in refmap:
        ref=np.load(d11[refmap[i],"orig"]["action"])[:,:6]
        out["to_control"]=[round(float(np.linalg.norm(a[:,:6]-ref)),3) for a in [ao,ae]]
    print(out)
set_mixed_pairs(pairs_small,pairs_extra,1.0,"lex_safe_full")
print("full compact payload",compact_embedding_payload())
```
