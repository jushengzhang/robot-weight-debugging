The check found a bookkeeping problem: `load_original` returns exactly the saved query-shift candidate for both adaptive-norm biases, despite an empty edit list. I will audit the other saved candidates and compare repeated original probes to determine whether this affected only the weight-analysis copy or also the forward-tool baseline. I will avoid further in-place operations on loaded originals.
```python
# Audit snapshot equality. No tensor returned by load_original is mutated.
audit=[]
for d in sorted(W.glob("candidate_*")):
    for p in sorted(d.glob("*")):
        n=p.name.rsplit(".",1)[0]
        if n not in NAMES:continue
        if p.stat().st_size>500_000_000:
            # Compare the explicitly edited rows only for large embedding tensors.
            if "lexical" in d.name and p.suffix==".npy":
                ids=torch.load(W/"subspaces"/"lexical_rows.pt",map_location="cpu",weights_only=False)["ids"]
                c=torch.from_numpy(np.array(np.load(p,mmap_mode="r")[ids])).float()
                a=load_original(n)[ids].clone()
                rel=float((a-c).norm()/a.norm())
            else:continue
        else:
            c=read_saved_tensor(p)
            if not torch.is_tensor(c):continue
            a=load_original(n)
            rel=float((a-c).norm()/(a.norm()+1e-12))
        audit.append((d.name,n,rel))
print("Exact matches:")
for d,n,r in audit:
    if r<1e-8:print(d,n)
print("Other FP32-related candidate differences:")
for d,n,r in audit:
    if any(t in n for t in ["embedding","dense.bias","linear.bias"]):
        print(d,n.split("model.")[-1],round(r,6))
# Repeated identical input paths from original probes.
orig_actions=collections.defaultdict(list)
for pp in sorted(W.glob("probe_*")):
    res=pp/"results.json"
    if not res.exists():continue
    for rr in json.loads(res.read_text()):
        if rr["arm"]=="orig":
            orig_actions[Path(rr["input"]).name+"|"+str(local_path(rr["input"]).parent.resolve())].append(
                (int(pp.name.split("_")[-1]),local_path(rr["action"])))
differences=[]
for key,seq in orig_actions.items():
    if len(seq)<2:continue
    seq.sort()
    a0=np.load(seq[0][1])
    for num,p in seq[1:]:
        err=float(np.max(np.abs(np.load(p)-a0)))
        differences.append((seq[0][0],num,err,key.split("|")[-1].split("/")[-1]))
print("Repeated original action comparisons",len(differences),
      "nonzero",sum(x[2]>0 for x in differences),
      "largest",sorted(differences,key=lambda x:x[2],reverse=True)[:8])
# Reconstruct clean-bias estimates from exact saved delta, but do not submit them.
clean_shift_bias={}
for l,s in qshift["shifts"].items():
    n=EP+f"layers.{l}.input_layernorm.dense.bias"
    a=load_original(n).clone()
    a[1024:2048]-=s.float()
    clean_shift_bias[n]=a
    print("Recovered shift RMS",l,float(a[1024:2048].square().mean().sqrt()))
torch.save(clean_shift_bias,W/"subspaces"/"recovered_preshift_biases.pt")
```
