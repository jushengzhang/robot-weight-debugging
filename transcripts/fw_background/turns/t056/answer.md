I will measure both the final action effect and the change along the edited activation directions. This checks whether the intervention reaches the intended component before spending rollout budget on it.
```python
s8=compare_probe(8)
(W/"probe8_summary.json").write_text(json.dumps(s8,indent=2))
# Compare light controls against their exact original probe references.
p8=read_probe(8)
def obskey(path):
    d=torch.load(path,map_location="cpu",weights_only=False)
    return str(d["suite"]),int(d["variant_id"]),int(d.get("step",0))
p7by={obskey(r["input"]):r for r in p7}
for j in range(8,12):
    row=p8[j]; ref=p7by[obskey(row["input"])]
    with np.load(row["activations"]) as ze, np.load(ref["activations"]) as zo:
        out=[]
        for L in [4,5,6]:
            key=f"mean::action.{L}.attn_out::action"
            d=ze[key]-zo[key]
            U=projection_data[f"U{L}"][:,:2]
            out.append((L,round(float(np.sqrt(np.mean(d*d))),5),
                        round(float(np.linalg.norm(U.T@d)),5)))
        print("Light activation shifts:",obskey(row["input"]),out)
print("Weight edit still active:",edited())
# Verify the affine anchor identity algebraically (no model execution).
for L in [4,5]:
    U=torch.from_numpy(projection_data[f"U{L}"][:,:2].copy())
    mu=torch.from_numpy(projection_data[f"mean{L}"].copy())
    stem=f"mot.mixtures.action.blocks.{L}.self_attn.o"
    b0=load_original(stem+".bias"); b1=load(stem+".bias")
    anchored=mu-0.35*U@(U.T@(mu-b0))+(b1-b0)
    print("L",L,"affine mean-anchor error",float((anchored-mu).abs().max()))
```