No text embeddings are saved in the forward artifacts. The two experts do share unusually large text-projection neurons, including action rows 571/934 and video rows 1714/2804. I will test whether these shared outliers amplify suffix changes by attenuating only those rows, while retaining their components along the known proprioception-encoder subspace.
```python
# Preserve the affine proprioception subspace in the 4096-dimensional context input.
Pw=load_original("proprio_encoder.weight")
Pb=load_original("proprio_encoder.bias")
Pbasis=torch.linalg.qr(torch.cat([Pw,Pb[:,None]],dim=1),mode="reduced").Q
outlier_rows={"action":[571,934],"video":[1714,2804]}
for e,rows in outlier_rows.items():
    W=load_original(f"mot.mixtures.{e}.text_embedding.0.weight")
    W2=load_original(f"mot.mixtures.{e}.text_embedding.2.weight")
    print(e, "row-pair cosine",round(float(torch.nn.functional.cosine_similarity(
        W[rows[0]][None],W[rows[1]][None])),4),
          "second-projection column norms",
          [round(float(W2[:,i].norm()),3) for i in rows],
          "proprio fractions",
          [round(float((W[i]@Pbasis).square().sum()/W[i].square().sum()),4) for i in rows])

def attenuate_text_rows(experts=("action","video"), scale=0.5, rows=outlier_rows):
    for e in experts:
        n=f"mot.mixtures.{e}.text_embedding.0.weight"
        W=load_original(n)
        new=W.clone()
        for i in rows[e]:
            w=W[i]
            protected=(w@Pbasis)@Pbasis.T
            new[i]=protected+scale*(w-protected)
        save_edit(n,new)

clear_edits()
attenuate_text_rows(scale=0.5)
config={"name":"text_outliers_2_half","rows":outlier_rows,
        "scale":0.5,"preserve":"proprio encoder affine span"}
(work/"current_edit.json").write_text(json.dumps(config,indent=2))
print("Current edits:",list(edited()))
with (work/"notes.md").open("a") as f:
    f.write("\n### Candidate text_outliers_2_half\n"
            "No text contexts found in saved artifacts. "
            "Action/video text projections share corresponding high-norm rows "
            "(action 571,934; video 1714,2804). Attenuate text-only components of those "
            "four rows by 0.5, exactly retaining components along proprio_encoder "
            "weight+bias span. Hypothesis: shared outlier nonlinear features amplify tag perturbations. "
            "Only two tensors edited; all earlier candidates cleared.\n")
```
