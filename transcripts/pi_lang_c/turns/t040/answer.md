The dairy backoff restores the milk task but loses several cheese-description recoveries, indicating that the equal category signal is too weak at half strength. I will test a full-strength category prototype, and analogous equal prototypes for the two meanings of “heating” and “holder.” These preserve ambiguity instead of forcing every occurrence to one object.
```python
rows_E=dict(rows_D); strengths_E=dict(strengths_D)
prototype_classes={"dairy":["cheese","milk","butter"],
                   "heating":["stove","microwave"],
                   "holder":["rack","basket"]}
for src,targets in prototype_classes.items():
    for pre in ["","\u2581"]:
        si=sp.piece_to_id(pre+src)
        if si==sp.unk_id() or sp.id_to_piece(si)!=pre+src or si in protected_ids:
            continue
        ti=[sp.piece_to_id(pre+t) for t in targets]
        assert all(t!=sp.unk_id() for t in ti)
        rows_E[si]=ti
        strengths_E[si]=1.0
set_prototype_edit(rows_E,strengths_E,"lexical_E")
print("E rows:",len(rows_E))
for i,t in rows_E.items():
    if isinstance(t,list):
        print(sp.id_to_piece(i),"=>", [sp.id_to_piece(j) for j in t],"strength",strengths_E[i])
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate E: full equal category prototypes\n"
            "D fixed 5/6 C validation regressions (remaining object1191 woven-holder phrasing), "
            "but lost dairy-specific and microwave recoveries. "
            "E tests full canonical-category centroids: dairy=mean(cheese,milk,butter), "
            "heating=mean(stove,microwave), holder=mean(rack,basket). "
            "No learned coefficients or target-label regression. Remaining D edits unchanged.\n")
```
