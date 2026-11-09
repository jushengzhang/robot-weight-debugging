BOS has over three times the norm of a typical content embedding. I’ll test a targeted lexical-residual edit on just 83 instruction-content rows: amplify each row’s component orthogonal to boundary/syntax embeddings, leaving BOS, newline, syntax, and the other 257,069 vocabulary rows unchanged. This tests lexical preservation rather than globally boosting language or actions.

```python
syntax_ids=sorted((all_prompt_ids-set(content_ids)-special) | {sp.bos_id()})
syntax_vectors=embedding[syntax_ids].T
syntax_basis=torch.linalg.qr(syntax_vectors,mode="reduced").Q
lexical_original=embedding[content_ids].clone()
lexical_component=lexical_original-(lexical_original@syntax_basis)@syntax_basis.T
print("syntax subspace dimension:",syntax_basis.shape[1],
      "lexical orthogonal energy fraction:",
      round(float(lexical_component.square().sum()/lexical_original.square().sum()),3))
def lexical_edit(gain=2.0,reset=True):
    if reset: clear_edits()
    e=embedding.clone()
    e[content_ids]=lexical_original+(gain-1)*lexical_component
    save_edit(EMB,e)
    changed=float((e[content_ids]-lexical_original).norm()/lexical_original.norm())
    del e
    return {"gain":gain,"rows":len(content_ids),"relative_changed_row_norm":round(changed,4)}
print("candidate:",lexical_edit(2.0))
snapshot("candidate_lexical_g2")
torch.save({"ids":content_ids,"syntax_ids":syntax_ids,"component":lexical_component},
           W/"subspaces"/"lexical_rows.pt")
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate lexical residual edit\n"
            "83 content-token embedding rows used by practice instructions; preserve boundary/syntax span. "
            "e'=e+(g-1)(I-Psyntax)e, g=2. BOS norm16.625 versus median content5.334. "
            "All other vocabulary rows unchanged. Test whether stronger lexical identity "
            "survives visual-context mixing and changes content attention/action choices.\n")
```
