I will summarize attention by valid instruction tokens, excluding padding and separating the BOS sink. I will also remove the shared flow-time component from expert residuals for analysis, so task-specific changes are not hidden by their large common offset.
```python
p31=probe_results(31)
print("ATTENTION METADATA",json.dumps(p31[0])[:3200])
literal_indices=[1,3,5,7,9]
print("CENTERED TASK-FEATURE COSINES")
for layer in [0,6,17]:
    key=f"mean::expert.{layer}.resid::action"
    center=np.mean([np.load(P30[(i,"orig")]["activations"])[key] for i in literal_indices],axis=0)
    vals=[]
    for i in range(0,12,2):
        orig=np.load(P30[(i,"orig")]["activations"])[key]-center
        edit=np.load(P30[(i,"edit")]["activations"])[key]-center
        target=np.load(P30[(i+1,"orig")]["activations"])[key]-center
        vals.append((probe_cases[i]["label"],round(cosine(orig,target),3),round(cosine(edit,target),3)))
    print("expert",layer,vals)
print("ATTN ARRAY",[(k,np.load(v).shape) for k,v in p31[0].get("attention",{}).items()] if all(
    isinstance(v,str) for v in p31[0].get("attention",{}).values()) else p31[0].get("attention"))
```
