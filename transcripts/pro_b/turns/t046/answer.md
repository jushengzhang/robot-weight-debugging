The probes provide pooled activations, not individual token states, so they cannot directly identify a deep-layer BOS direction. I will use them only for comparisons they support: instruction sensitivity across layers, attention concentration by head, and unusually dominant weight directions. This should narrow the next causal intervention without treating pooled features as exact token representations.
```python
a3 = np.load(pr[3][0]["activations"])
a13 = np.load(pr[13][0]["activations"])
print("Activation keys:", a3.files[:24], a3.files[-24:])
print("MLP keys:", a13.files[100:116], a13.files[-10:])
for k in [2,5,10]:
    print("\nAttention probe", k)
    for x in pr[k]:
        if x["arm"] != "orig": continue
        nv = x["prefix_layout"]["n_lang_valid"]
        print("input",x["idx"], Path(x["input"]).parent.name)
        for l, p in x["attention"].items():
            a = np.load(p).mean(axis=1)
            bos = a[:,768]
            content = a[:,769:768+nv-1].sum(axis=1)
            wrist = a[:,256:512].sum(axis=1)
            peak = a[:,:512].max(axis=1)
            print(" L"+l, "BOS",np.round(bos,2).tolist(),
                  "word",np.round(content,2).tolist(),
                  "wrist",np.round(wrist,2).tolist())
        if x["idx"] >= 1: break
lm = "paligemma_with_expert.paligemma.model.language_model"
ex = "paligemma_with_expert.gemma_expert.model"
print("\nVLM norm statistics (of 1+weight):")
for l in range(18):
    g = 1 + load_original(f"{lm}.layers.{l}.input_layernorm.weight")
    v, i = torch.topk(g.abs(),4)
    print(l, "median/max",round(g.abs().median().item(),3),round(v[0].item(),3),
          "top",i.tolist())
print("\nExpert Q singular concentration:")
for l in range(18):
    q = load_original(f"{ex}.layers.{l}.self_attn.q_proj.weight").reshape(8,256,1024)
    s = torch.linalg.svdvals(q)
    frac = (s[:,0]**2 / s.square().sum(1))
    print(l, [round(z,3) for z in frac.tolist()])
```
