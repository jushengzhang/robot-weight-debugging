The earlier instruction edits did not alter behavior as predicted, so I will check the diagnostic tensors more carefully before designing another one. In particular, I want to distinguish attention to actual instruction words from attention to BOS, and determine which quantities the saved activations really expose.
```python
torch.set_num_threads(min(16, os.cpu_count() or 16))
pr = {}
for k in [2,3,5,6,10,13]:
    pr[k] = json.loads((W/f"probe_{k}/results.json").read_text())
    z = pr[k][0]
    print("probe", k, "n",len(pr[k]),"keys", list(z))
    if "activations" in z:
        with np.load(z["activations"]) as a:
            print("  arrays:", [(key,a[key].shape) for key in a.files[:8]], "total",len(a.files))
    if "attention" in z:
        print("  attention:", [(key,np.load(p).shape) for key,p in z["attention"].items()])
print("subspaces:", [p.name for p in (W/"subspaces").iterdir()])
print("frame listing:", [p.name for p in Path(r1[0]["frames_dir"]).iterdir()][:8])
print("key tensors:")
for n in NAMES:
    if ("language_model.layers.6." in n or "gemma_expert.model.layers.6." in n
        or ("norm" in n and "layers" not in n) or "action_" in n):
        print(n, SHAPES[n])
print("probe13 example metadata:", pr[13][0])
print("split sizes:", {k:len(v) for k,v in json.loads((W/"split_manifest.json").read_text()).items()})
```
