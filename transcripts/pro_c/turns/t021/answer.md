I will compare BOS and content-token attention before and after the key-column edit, along with action changes. This is a causal test of the proposed routing mechanism; a behavioral gain alone would not show that the intended pathway changed.
```python
def summarize_probe(k):
    pairs=probe_pairs(k)
    for i,pair in pairs.items():
        o,e=pair["orig"],pair["edit"]
        ids,text=prompt_tokens(o["input"])
        print("case",i,text[:60])
        for l in o.get("attention",{}):
            vals=[]
            for pp in [o,e]:
                a=np.load(pp["attention"][l]).astype(np.float32).mean(1)
                vals.append([100*a[:,768].mean(),100*a[:,769:768+len(ids)-1].sum(-1).mean()])
            print(" L"+l, "BOS/content orig->edit", [[round(float(x),2) for x in v] for v in vals])
        a,b=np.load(o["action"]),np.load(e["action"])
        print(" action delta RMS",np.round(np.sqrt(((b-a)**2).mean(0)),4).tolist())
summarize_probe(7)
```
