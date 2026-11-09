I will measure the attention change and action displacement from this intervention. If reducing BOS does not increase useful instruction access, or distorts all actions indiscriminately, I will reject the hypothesis rather than treat a sink as automatically harmful.
```python
def probe_pairs(k):
    data=json.loads((W/f"probe_{k}/results.json").read_text())
    out=collections.defaultdict(dict)
    for p in data: out[p["idx"]][p["arm"]]=p
    return out
pairs5=probe_pairs(5)
print("arms:",list(pairs5[0]))
for i,pair in pairs5.items():
    orig=pair.get("orig",pair.get("original"))
    ed=pair.get("edit",pair.get("edited"))
    ids,text=prompt_tokens(orig["input"])
    print("input",i,text)
    for l in orig["attention"]:
        vals=[]
        for pp in [orig,ed]:
            a=np.load(pp["attention"][l]).astype(np.float32).mean(1)
            vals.append([round(float(100*x),2) for x in
                         [a[:,768].mean(),a[:,769:768+len(ids)-1].sum(-1).mean(),
                          a[:,:256].sum(-1).mean(),a[:,256:512].sum(-1).mean()]])
        print(" L",l,"BOS/content/base/wrist:",vals)
    a=np.load(orig["action"]); b=np.load(ed["action"])
    print(" action RMS change xyz/rot/grip:",
          [round(float(np.sqrt(np.mean((a[:,s]-b[:,s])**2))),4) for s in [slice(0,3),slice(3,6),slice(6,7)]])
```
