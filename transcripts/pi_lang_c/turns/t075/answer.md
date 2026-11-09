After centering the shared component, the repaired expert features align much more closely with the intended tasks—for example, the cheese case’s layer-17 cosine rises from -0.039 to 0.857. I will now test a half-strength version as a temporary causal ablation, then restore the already-confirmed full recipe.
```python
P31={(r["idx"],r["arm"]):r for r in p31}
for i,case_idx in enumerate([0,2,4,6]):
    prompt=probe_cases[case_idx]["instruction"]
    ids=sp.encode(prompt+"\n",add_bos=True)
    lay=P31[(i,"orig")]["prefix_layout"]
    start=lay["lang"][0]
    nv=lay["n_lang_valid"]
    assert len(ids)==nv,(prompt,len(ids),nv)
    edited_positions=[start+j for j,t in enumerate(ids) if t in F["rows"]]
    print("ATTENTION",probe_cases[case_idx]["label"],"edited tokens",[(j,ascii_piece(t)) for j,t in enumerate(ids) if t in F["rows"]])
    for layer in [0,3,6,12]:
        arrs=[np.load(P31[(i,arm)]["attention"][str(layer)]) for arm in ["orig","edit"]]
        row={"L":layer}
        row["text_no_bos"]=[round(float(a[:,:,start+1:start+nv].sum(-1).mean()),3) for a in arrs]
        row["bos"]=[round(float(a[:,:,start].mean()),3) for a in arrs]
        row["edited_tokens"]=[round(float(a[:,:,edited_positions].sum(-1).mean()),3) for a in arrs]
        h=arrs[0][:,:,start+1:start+nv].sum(-1).mean(-1)
        row["original_top_text_heads"]=np.argsort(-h)[:2].tolist()
        print(row)
Fhalf=copy.deepcopy(F)
Fhalf["strengths"]={i:0.5*a for i,a in F["strengths"].items()}
install_recipe(Fhalf,"F_half_causal")
with (W/"notes.md").open("a") as f:
    f.write("\n## Centered task features and planned dose test\nPass30 expert residuals share a large time component. "
            "Subtract mean of5 literal-task controls for analysis only. Layer17 target cosine original->F: "
            "cooker .462->.975; basin .154->.843; grape .006->.831; spread -.039->.857; slide .513->.561. "
            "Attention maps pass31 test whether language was already read. "
            "Temporarily installed F_half_causal (each sparse interpolation coefficient halved) for probe32 "
            "and diagnostic rollout33. FINAL CHOICE REMAINS F; restore before rollout34 and remaining validation.\n")
```
