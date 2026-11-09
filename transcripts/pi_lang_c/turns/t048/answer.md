The condiment regression now depends only on the “lift” row, so I will restore both lift embeddings. For the dairy failures, I will preserve the broad dairy centroid but add a separate spread-category centroid and the more specific “churn → butter” cue. This should distinguish spreads from liquid milk without forcing every dairy reference toward cheese.
```python
F = copy.deepcopy(E)
for i in [20252,12758]:
    F["rows"].pop(i,None)
    F["strengths"].pop(i,None)
for lead in ["","\u2581"]:
    src=sp.piece_to_id(lead+"spread")
    targets=[sp.piece_to_id(lead+w) for w in ["cream","cheese","butter"]]
    assert src != sp.unk_id() and all(t != sp.unk_id() for t in targets)
    F["rows"][src]=targets
    F["strengths"][src]=0.75
    src=sp.piece_to_id(lead+"churn")
    dst=sp.piece_to_id(lead+"butter")
    if src != sp.unk_id() and src not in protected:
        F["rows"][src]=dst
        F["strengths"][src]=0.75
install_recipe(F,"lexical_F_compositional")
print("Added rows",[(ascii_piece(i),v,F["strengths"][i]) for i,v in F["rows"].items() if i not in E["rows"]])
print("1191 touched",[(i,ascii_piece(i)) for i in sp.encode(reg["instruction"]) if i in F["rows"]])
with (W/"notes.md").open("a") as f:
    f.write("\n## E ablation (pass 18)\nD and E identical success outcomes on all 24 cases. "
            "Remaining regression object1191 has exactly one edited instruction token: initial lift->pick. "
            "Restore lift rows; dropping woven did not lose any gains. "
            "F adds spread->centroid(cream,cheese,butter) at .75 and churn->butter at .75, "
            "retaining dairy->centroid(cheese,milk,butter) at .5. This separates category and distinguishing modifier.\n")
```
