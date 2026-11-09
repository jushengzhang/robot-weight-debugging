I will inspect the probe tensors and define a small, reversible lexical intervention. The first test will change only noncanonical embedding rows for stove-related and wine-related cues; all action, vision, and canonical-instruction weights will remain untouched.
```python
p5=json.loads((W/"probe_5"/"results.json").read_text())
print("probe fields", list(p5[0]))
print("activation path",p5[0].get("activations"))
ap5 = [dict(np.load(p["activations"])) for p in p5]
print("activation shapes",[(k,v.shape) for k,v in ap5[0].items()][:25])
# Inspect common embedding direction without evaluating the network.
emean=E0.mean(0)
print("mean embedding norm",round(emean.norm().item(),3))
for s,t in [("cook","stove"),("cooker","stove"),("grape","wine"),("grapes","wine"),("fermented","wine"),("cupboard","cabinet"),("dairy","cheese"),("citrus","orange")]:
    si=sp.encode("the "+s)[1:];ti=sp.encode("the "+t)[1:]
    if len(si)==len(ti)==1:
        a,b=E0[si[0]]-emean,E0[ti[0]]-emean
        print(s,t,si[0],ti[0],"centered cos",round(float(torch.nn.functional.cosine_similarity(a,b,dim=0)),3),
              "centered norms",round(a.norm().item(),2),round(b.norm().item(),2))
# Select a strictly semantic, two-family diagnostic edit.
lex_small={"cook":"stove","cooker":"stove","grape":"wine","grapes":"wine","fermented":"wine"}
def single_word_id(s):
    ids=sp.encode("the "+s)[1:]
    assert len(ids)==1,(s,ids)
    return ids[0]
pairs_small={single_word_id(s):single_word_id(t) for s,t in lex_small.items()}
assert not (set(pairs_small)&protected_ids)
def set_embedding_pairs(pairs,alpha=1.0,tag=None):
    e=E0.clone()
    for s,t in pairs.items():
        e[s]=E0[s]+alpha*(E0[t]-E0[s])
    save_edit(Ename,e)
    del e
    if tag:
        (W/(tag+"_spec.json")).write_text(json.dumps({"alpha":alpha,"pairs":{str(s):int(t) for s,t in pairs.items()}},indent=2))
set_embedding_pairs(pairs_small,0.5,"lex_small_half")
print("edited rows",[(piece_ascii(s),piece_ascii(t)) for s,t in pairs_small.items()])
with (W/"notes.md").open("a") as f:
    f.write("\nPass 2 easy goal controls: 17/18 (94.4%). Pass 4 broad language: 107/128 (83.6%), "
            "goal 22/34 vs other suites 85/94. Attention passes 3/5: language attended in failures, "
            "BOS/newline are major hubs; not evidence for globally increasing language attention.\n"
            "First causal edit: 5 noncanonical lexical rows (cook/cooker -> stove; grape/grapes/fermented -> wine), "
            "50% displacement toward the canonical row. All 74 canonical task token IDs protected.\n")
```
