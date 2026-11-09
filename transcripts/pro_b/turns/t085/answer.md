I will finish the head-level check with the actual attention-map shape. I will also compare the object-name subspace with the projector’s output energy; weak transmission there is only a hypothesis, not yet evidence that amplifying it would help.
```python
for i in range(4):
    a=np.load(W/"probe_21"/f"probe_orig_{i}_attn_L14.npy")
    b=np.load(W/"probe_21"/f"probe_edit_{i}_attn_L14.npy")
    print("input",i)
    for h in range(8):
        x=a[h].mean(axis=0); y=b[h].mean(axis=0)
        peaks=np.argsort(x[:512])[-2:][::-1]
        print(h,"vis",round(float(x[:512].sum()),3),
              "outliers",round(float(x[outlier].sum()+x[np.array(outlier)+256].sum()),3),
              "TV",round(float(np.abs(x[:512]-y[:512]).sum()/2),3),
              "peaks",[(int(j),round(float(x[j]),3)) for j in peaks])
proj_name=[n for n in NAMES if "multi_modal_projector.linear.weight" in n][0]
M=load_original(proj_name)
U=sobj["U"].float()
print("Object output energy fraction",float((U.T@M).square().sum()/M.square().sum()),
      "isotropic fraction",16/2048)
print("Overlap positional outliers with canonical position mean:")
Pc=P-P.mean(0)
print([(i,round(float(torch.cosine_similarity(P[i],P.mean(0),dim=0)),3)) for i in outlier])
print("Task diagnostic request structure",type(json.loads((W/"lex_test_ids.json").read_text())))
print("First saved-input keys:")
inp_list=json.loads((W/"semantic_swap_inputs.json").read_text())
def local_path(s):
    s=str(s)
    marker="/work/"
    return W/s.split(marker,1)[1] if marker in s else Path(s)
inp=torch.load(local_path(inp_list[0]),map_location="cpu",weights_only=False)
print([(k,tuple(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in inp.items()])
```
