The regressions expose specific overgeneralizations: “dairy” was forced to cheese even when it meant milk, “heating” to microwave when it meant stove, and “holder” to rack when it meant basket. I will remove those hard sense assignments, preserve “dairy” only as an equal cheese/milk/butter category prototype, and weaken broad shape/material cues. This is a semantic backoff, not a fitted correction to individual task labels.
```python
# D: back off polysemous aliases. Equal category prototype for dairy; no task-label fitting.
remove_words={"heating","holder","ceramic","shut"}
def bare_piece(i): return sp.id_to_piece(i).lstrip("\u2581")
rows_D={i:j for i,j in rows_C.items() if bare_piece(i) not in remove_words}
strengths_D={i:strengths_C[i] for i in rows_D}
for i in rows_D:
    word=bare_piece(i)
    if word=="woven": strengths_D[i]=.25
    if word in {"round","flat"}: strengths_D[i]=.5
    if word=="dairy":
        pre="\u2581" if sp.id_to_piece(i).startswith("\u2581") else ""
        targets=[sp.piece_to_id(pre+t) for t in ["cheese","milk","butter"]]
        assert all(t!=sp.unk_id() for t in targets)
        rows_D[i]=targets
        strengths_D[i]=.5
def set_prototype_edit(rows,strengths,recipe_name=None):
    for n in list(edited()): drop_edit(n)
    E=E0.clone()
    for si,target in rows.items():
        tv=E0[target].mean(0) if isinstance(target,(list,tuple)) else E0[target]
        a=float(strengths.get(si,1.0))
        E[si]=E0[si]+a*(tv-E0[si])
    save_edit(EMB,E)
    del E
    if recipe_name:
        torch.save({"rows":rows,"strengths":strengths},W/(recipe_name+"_recipe.pt"))
set_prototype_edit(rows_D,strengths_D,"lexical_D")
print("Candidate D rows:",len(rows_D),"canonical overlap",len(set(rows_D)&protected_ids))
print("Removed pieces:",[sp.id_to_piece(i) for i in rows_C if i not in rows_D])
print("Dairy targets:",[(sp.id_to_piece(i),[sp.id_to_piece(t) for t in j]) for i,j in rows_D.items() if isinstance(j,list)])
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate D: semantic backoff after C failed confirmation\n"
            "Remove heating->microwave, holder->rack, ceramic->mug, shut->close. "
            "Dairy becomes .5 original + .5 equal centroid(cheese,milk,butter), not cheese-specific. "
            "Woven->basket reduced to .25; round->bowl and flat->plate to .5. "
            f"{len(rows_D)} rows, all canonical tokens still untouched. "
            "Will test known regressions and dairy-positive controls, then untouched confirmation 2.\n")
```
