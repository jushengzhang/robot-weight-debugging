Layer 4 contains a useful contrast: heads 0 and 4 are image/BOS dominated, while heads 3 and 7 read actual instruction words. I will inspect which words those heads read before choosing a head-specific intervention; strengthening a destination preposition is not equivalent to strengthening target-object selection.
```python
for p in [p2[1],p4[1],p4[3]]:
    ids,text=prompt_tokens(p["input"])
    print(text)
    for l,path in p["attention"].items():
        if int(l) not in [4,6,8,10,12,16]: continue
        a=np.load(path).astype(np.float32).mean(1)[:,768:768+len(ids)]
        print("L"+l)
        for h in range(8):
            ix=np.argsort(a[h,1:-1])[-3:][::-1]+1
            if a[h,1:-1].sum()>.05:
                print(" h",h,"content",round(float(a[h,1:-1].sum()*100),1),
                      [(sp.id_to_piece(ids[j]).replace("\u2581","_"),round(float(a[h,j]*100),1)) for j in ix])
# Set up reproducible head-block interventions.
def set_head_edit(scales):
    clear_edits()
    for l,hs in scales.items():
        n=f"{EP}.layers.{l}.self_attn.o_proj.weight"
        w=load_original(n)
        for h,factor in hs.items():
            w[:,h*256:(h+1)*256] *= factor
        save_edit(n,w)
print("All edits now:",len(edited()))
```
