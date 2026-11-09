Some tempting mappings are unsafe: “vessel” also denotes a mug, and “dairy” can denote milk or butter. I will leave those rows untouched. The next controlled extension uses narrower anchors—rounded/mixing bowl cues, citrus, rack framework, spatial superlatives, and activation verbs—and tests them at half strength first.
```python
pairs_extra={}
lex_extra={"rounded":"bowl","mixing":"bowl","citrus":"orange","framework":"rack",
           "uppermost":"top","forefront":"front","forward":"front","cultured":"cream"}
for s,t in lex_extra.items():
    pairs_extra[single_word_id(s)]=single_word_id(t)
for s,t in [("activate","turn"),("ignite","turn")]:
    for space in [False,True]:
        si=sp.encode(("the " if space else "")+s)[1 if space else 0:]
        ti=sp.encode(("the " if space else "")+t)[1 if space else 0:]
        if len(si)==len(ti)==1:
            pairs_extra[si[0]]=ti[0]
assert not(set(pairs_extra)&protected_ids)
def set_mixed_pairs(fixed, trial, alpha, tag=None):
    e=E0.clone()
    for s,t in fixed.items(): e[s]=E0[t]
    for s,t in trial.items(): e[s]=E0[s]+alpha*(E0[t]-E0[s])
    save_edit(Ename,e)
    del e
    if tag:
        (W/(tag+"_spec.json")).write_text(json.dumps({"fixed":fixed,"trial":trial,"alpha":alpha},indent=2))
set_mixed_pairs(pairs_small,pairs_extra,.5,"lex_safe_half")
print("extra rows",[(s,piece_ascii(s),piece_ascii(t)) for s,t in pairs_extra.items()])
# Use only already-seen practice observations for the next causal panel.
probe_safe_records=[]
for suite,vid in [("libero_goal",1415),("libero_goal",1416),("libero_goal",1400),("libero_goal",1390),
                  ("libero_goal",1179),("libero_goal",1169),("libero_goal",1339),("libero_goal",1341),
                  ("libero_goal",1378),("libero_goal",1263),("libero_goal",1278),("libero_goal",1431),
                  ("libero_goal",1331),("libero_goal",1462),("libero_object",1376),("libero_object",1377)]:
    probe_safe_records.append(next(r for r in r1+r2+r4 if r["suite"]==suite and r["id"]==vid))
print("probe paths")
print(json.dumps([first_input(r).replace(str(W),"WORK") for r in probe_safe_records]))
```
