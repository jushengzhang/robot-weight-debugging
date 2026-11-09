Only two early MLP directions pass the reproducibility screen; the later failure directions remain task-dependent. Before making another downstream edit, I’ll inspect the saved observations for a shared spatial-frequency disturbance at the visual input. If the disturbance is concentrated in a separable patch-embedding mode, that offers a more direct intervention before texture information mixes with task features.
```python
sample_input = torch.load(manifest19["source_paths"][0],
                          map_location="cpu",weights_only=False)
def compact_schema(x,depth=0):
    if torch.is_tensor(x) or isinstance(x,np.ndarray):
        return {"type":type(x).__name__,"shape":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x,dict) and depth<3:
        return {str(k):compact_schema(v,depth+1) for k,v in x.items()}
    if isinstance(x,(list,tuple)) and depth<2:
        return [compact_schema(v,depth+1) for v in x[:3]]
    return str(x)[:140]
print("Policy input schema:",json.dumps(compact_schema(sample_input),indent=2))
print("\nVideo patch embedding tensors:")
for n in NAMES:
    if "video" in n and ("patch" in n):
        print(n,SHAPES[n])
print("\nMLP conclusion: shared late failure direction not established; no MLP edit applied.")
print("Current edits:",edited())
```
