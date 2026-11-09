The query shift changes word attention but barely changes actions (RMS differences below 0.002), so these mid-layer heads appear weakly causal for the initial decision. I will revert it rather than spend a rollout. Before probing earlier layers, I will verify that the matched task and semantic observations really share identical images; otherwise instruction contrasts could be confounded by visual differences.
```python
for n in list(edited()): drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 19: query-content shift not pursued\n"
            "Word attention increases (L6H2 tomato .075->.121; L12H2 wine .252->.357), "
            "but action RMS differences only .0004-.0016. No rollout; reverted. "
            "Mid-layer word-reading heads may not control the initial action choice.\n")
p0=torch.load(pr[2][0]["input"],map_location="cpu",weights_only=False)
p1=torch.load(pr[2][1]["input"],map_location="cpu",weights_only=False)
def tree_info(x, prefix=""):
    if isinstance(x,dict):
        for k,v in x.items(): tree_info(v,prefix+"/"+str(k))
    elif isinstance(x,(torch.Tensor,np.ndarray)):
        print(prefix,tuple(x.shape),str(x.dtype))
    else:
        print(prefix,type(x).__name__,str(x)[:100])
tree_info(p0)
def tensor_diff(a,b,prefix=""):
    if isinstance(a,dict) and isinstance(b,dict):
        for k in a.keys() & b.keys(): tensor_diff(a[k],b[k],prefix+"/"+str(k))
    elif isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.shape==b.shape:
        if a.dtype==torch.bool: 
            print(prefix,"different fraction",float((a!=b).float().mean()))
        else:
            d=(a.float()-b.float())
            print(prefix,"RMS",round(float(d.square().mean().sqrt()),6),
                  "max",round(float(d.abs().max()),6))
    elif isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.shape==b.shape:
        d=a.astype(float)-b.astype(float)
        print(prefix,"RMS",round(float(np.sqrt((d*d).mean())),6))
print("\nTask-semantic input differences:")
tensor_diff(p0,p1)
print("\nAvailable early-layer act arrays:")
print([(k,a3[k].shape) for k in a3.files if k.startswith("calls::expert")][:3])
```
