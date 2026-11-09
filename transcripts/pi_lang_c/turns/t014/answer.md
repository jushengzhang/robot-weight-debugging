The synonym embeddings are close in raw cosine but differ substantially in norm, and the expert’s task-centered features are weaker or misrouted on several failures. I will test a narrow causal intervention: move only unambiguous synonym rows halfway toward their canonical object-name rows, while leaving every canonical instruction token untouched.
```python
Emean=E0.mean(dim=0)
print("Embedding mean norm:",round(float(Emean.norm()),4))
for a,b in [("cooker","stove"),("basin","bowl"),("cupboard","cabinet"),("grape","wine")]:
    ia=sp.piece_to_id("\u2581"+a); ib=sp.piece_to_id("\u2581"+b)
    print(a,b,"centered cosine",round(float(torch.nn.functional.cosine_similarity(E0[ia]-Emean,E0[ib]-Emean,dim=0)),3))
base_aliases={"cooker":"stove","basin":"bowl","cupboard":"cabinet",
              "grape":"wine","grapes":"wine","fermented":"wine","merlot":"wine"}
def alias_rows(aliases):
    rows={}
    for src,tgt in aliases.items():
        for pre in ["","\u2581"]:
            si=sp.piece_to_id(pre+src); ti=sp.piece_to_id(pre+tgt)
            if si==sp.unk_id() or sp.id_to_piece(si)!=pre+src:
                continue
            if ti==sp.unk_id() or sp.id_to_piece(ti)!=pre+tgt:
                tids=sp.encode(tgt) if not pre else sp.encode("the "+tgt)[1:]
                if len(tids)!=1: continue
                ti=tids[0]
            if si in protected_ids:
                print("PROTECTED, skipped",src,si)
                continue
            rows[si]=ti
    return rows
base_rows=alias_rows(base_aliases)
def set_alias_edit(rows, alpha=1.0, label=None):
    for n in list(edited()):
        drop_edit(n)
    E=E0.clone()
    si=torch.tensor(sorted(rows),dtype=torch.long)
    ti=torch.tensor([rows[int(i)] for i in si],dtype=torch.long)
    E[si]=E0[si]+alpha*(E0[ti]-E0[si])
    save_edit(EMB,E)
    if label:
        torch.save({"rows":rows,"alpha":alpha,"aliases":base_aliases if rows==base_rows else None},
                   W/(label+"_recipe.pt"))
    del E
set_alias_edit(base_rows,0.5,"lexical_base_half")
print("Edited rows:",[(i,sp.id_to_piece(i),sp.id_to_piece(j)) for i,j in base_rows.items()])
print("Edited tensors:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate A: sparse unambiguous lexical aliases\n"
            "cooker->stove, basin->bowl, cupboard->cabinet, grape/grapes/fermented/merlot->wine. "
            f"{len(base_rows)} embedding rows; all 76 canonical task token IDs protected. "
            "First causal dose alpha=0.5. No other tensor changes.\n")
```
