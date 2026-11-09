I will separate content-word, BOS, final-token, and visual attention in the early layers, rather than counting all language tokens together. This will show whether an early head is a plausible instruction route or another sink-reading pathway.
```python
p9=json.loads((W/"probe_9/results.json").read_text())
def rounded(x,dec=1):
    return [round(float(v),dec) for v in np.asarray(x).ravel()]
def summarize_attention(prs, layers, only_original=False):
    for pr in prs:
        if only_original and pr["arm"]!="orig": continue
        nv=pr["prefix_layout"]["n_lang_valid"]
        print("CASE",pr["idx"],pr["arm"])
        for l in layers:
            m=np.load(pr["attention"][str(l)]).mean(1).astype(np.float64)
            print(l,"base",rounded(100*m[:,:256].sum(1)),
                  "wrist",rounded(100*m[:,256:512].sum(1)),
                  "BOS",rounded(100*m[:,768]),
                  "words",rounded(100*m[:,769:768+nv-1].sum(1)),
                  "end",rounded(100*m[:,768+nv-1]))
summarize_attention(p9,[0,1,2,4])
print("Early head top tokens (case0):")
z=torch.load(W/"roll_8_frames/e1934_seed7_original/t0000_inputs.pt",map_location="cpu",weights_only=False)
ids=prompt_ids(z["element"]["prompt"])
for l in [0,1,2,4]:
    m=np.load(p9[0]["attention"][str(l)]).mean(1)
    print("L",l,[
        [(int(j),round(float(m[h,j]),3)) for j in np.argsort(m[h])[-3:][::-1]]
        for h in range(8)])
# Architecture and norm information for interpreting direct value paths.
print("Early expert adaptive normalization bias partition RMS:")
for l in [0,1,2,4,9,14,16]:
    b=load_original(f"{expert_root}.layers.{l}.input_layernorm.dense.bias").reshape(3,1024)
    print(l,rounded(b.square().mean(1).sqrt(),4))
```
