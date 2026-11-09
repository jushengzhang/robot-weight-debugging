I will summarize the heads that attend to the repaired words, then add the verb aliases at 75% strength. The extension is deliberately separate: “slide” to “push,” “shut/secure” to “close,” and similar action synonyms should help the remaining primitive-selection failures without changing literal commands.
```python
e13=entries(read_probe(13))
for i in [1420,1262,1179,1360]:
    eo=next(e for e in e13 if input_id(e)==i and e["arm"]=="orig")
    ee=next(e for e in e13 if input_id(e)==i and e["arm"]!="orig")
    dd=torch.load(eo["input"],map_location="cpu",weights_only=False)
    ids=[sp.bos_id()]+sp.encode(dd["element"]["prompt"]+"\n")
    hit=[j for j,t in enumerate(ids) if t in candidate_rows]
    print("ID",i,"edited pieces",[(j,sp.id_to_piece(ids[j])) for j in hit])
    for l in [0,3,6,12]:
        ao=np.load(eo["attention"][str(l)]).mean(1)
        ae=np.load(ee["attention"][str(l)]).mean(1)
        mo=ao[:,np.array(hit)+768].sum(1); me=ae[:,np.array(hit)+768].sum(1)
        hs=np.argsort(np.maximum(mo,me))[-3:][::-1]
        print(" L",l,[(int(h),round(float(mo[h]),4),round(float(me[h]),4)) for h in hs])
rows_C={**candidate_rows,**verb_rows}
strengths_C={**strengths_B,**{i:.75 for i in verb_rows}}
set_mixed_alias_edit(rows_C,strengths_C,"lexical_C")
print("Candidate C:",len(rows_C),"rows; canonical-token overlap",len(set(rows_C)&protected_ids))
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate C\nAdds 25 verb-synonym embedding rows at .75 to candidate B. "
            "Motivation: residual slide/push, pull/open, secure/close primitive confusions. "
            "Canonical tokens remain protected; no action/expert weights changed.\n")
```
