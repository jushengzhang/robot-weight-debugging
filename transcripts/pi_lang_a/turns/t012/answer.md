The expert is not ignoring language: several heads attend strongly to it, especially the newline summary token. I will test a narrower hypothesis—synonym embeddings do not trigger the learned bowl/cabinet task code reliably—by moving only the “basin” and “cupboard” rows halfway toward their literal counterparts. All other rows and all downstream weights will remain unchanged.

```python
Emean = E0.mean(0)
print("global mean norm:",round(float(Emean.norm()),3))
for s,t in [("basin","bowl"),("cupboard","cabinet"),("cooker","stove"),("dish","plate"),
            ("rounded","bowl"),("compartment","drawer"),("fermented","wine"),("bowl","stove")]:
    a=E0[word_ids(s)].mean(0)-Emean
    b=E0[word_ids(t)].mean(0)-Emean
    print(s,t,"centered cos",round(float(torch.nn.functional.cosine_similarity(a,b,dim=0)),3))
canonical_texts = sorted({
    re.sub(r"^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", v["name"].split("_language_")[0]).replace("_"," ").lower()
    for v in variants if v["category"]=="Language Instructions"
})
canonical_ids = set()
for text in canonical_texts:
    canonical_ids.update(sp.encode(text+"\n",out_type=int,add_bos=True))
print("canonical tasks:",len(canonical_texts),"protected token IDs:",len(canonical_ids))
print("canonical sample:",canonical_texts[:5], canonical_texts[-5:])
alias_pair = {"basin":"bowl","cupboard":"cabinet"}
def apply_aliases(mapping, alpha=1.0, tag="candidate"):
    E = E0.clone()
    metadata=[]
    for source, target in mapping.items():
        si,ti=word_ids(source),word_ids(target)
        assert len(si)==1 and len(ti)==1, (source,si,target,ti)
        assert si[0] not in canonical_ids, ("protected",source,si[0])
        E[si[0]] = (1-alpha)*E0[si[0]] + alpha*E0[ti[0]]
        metadata.append({"source":source,"target":target,"source_id":si[0],"target_id":ti[0],"alpha":alpha})
    save_edit(emb_name,E)
    del E
    (W/f"{tag}.json").write_text(json.dumps(metadata,indent=2))
    return metadata
pair_half = apply_aliases(alias_pair,0.5,"pair_half")
print("edited rows:",pair_half)
print("edited tensors:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\nPass3 attention: language is not absent. L3 heads1/4 attend chiefly BOS; "
            "L6 head6 and L9 head4 read newline summary (newline mass roughly0.56-0.62 for plate/stove pair). "
            "Synonyms change initial actions despite strong text attention. "
            "Causal test A: only space-prefixed basin->bowl and cupboard->cabinet embedding rows, alpha0.5. "
            "Both are absent from canonical task-token set; no downstream edits.\n")
```
