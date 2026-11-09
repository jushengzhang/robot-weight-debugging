The added bowl cues create one overcorrection when both “rounded” and “mixing” occur together, so I will reduce those two displacements. I will separately test destination descriptors—storage “unit,” “furniture,” and “structure”—at half strength. These are more ambiguous than the first aliases, so they remain provisional and must pass broad and independent validation.
```python
spec_base={s:(t,1.0) for s,t in {**pairs_small,**pairs_extra}.items()}
for word in ["rounded","mixing"]:
    s=single_word_id(word); spec_base[s]=(spec_base[s][0],.5)
pairs_dest={single_word_id(s):single_word_id(t) for s,t in {"unit":"cabinet","furniture":"cabinet","structure":"rack"}.items()}
def apply_spec(spec,tag=None,compact=True):
    assert not(set(spec)&protected_ids)
    e=E0.clone()
    for s,(t,a) in spec.items():
        e[s]=E0[s]+float(a)*(E0[t]-E0[s])
    save_edit(Ename,e)
    del e
    if compact: compact_embedding_payload()
    if tag:
        (W/(tag+"_spec.json")).write_text(json.dumps({str(s):[int(t),float(a)] for s,(t,a) in spec.items()},indent=2))
spec_dest_half={**spec_base,**{s:(t,.5) for s,t in pairs_dest.items()}}
apply_spec(spec_dest_half,"lex_dest_half")
print("candidate rows",len(spec_dest_half))
print("destination additions",[(piece_ascii(s),piece_ascii(t)) for s,t in pairs_dest.items()])
print("half bowl cues",[(piece_ascii(s),a) for s,(t,a) in spec_dest_half.items() if a<1])
```
