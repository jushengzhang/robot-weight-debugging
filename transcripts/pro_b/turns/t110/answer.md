Neither singleton head attenuation supports a reliable repair, so I will restore both. My last new candidate family will target the vision encoder’s five extreme positional rows, which the earlier smooth-position edit deliberately left untouched. I will reduce only their excess over neighboring positions, testing whether excessively position-dominated register tokens impair upstream object localization while preserving ordinary patch positions.
```python
for n in list(edited()):drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass33: isolated L3H3 attenuation rejected\n"
            "60/128->58/128 (-1.6 +/-1.6pp),1win/3losses,task20->18,environment7->6. "
            "No isolated head supports the earlier apparent loop benefit. Restored original.\n")
P0=original_clone(pn)
norms=P0.norm(dim=1)
rows=torch.where(norms>3*norms.median())[0].tolist()
neighbor={}
excess=torch.zeros_like(P0)
for i in rows:
    r,c=divmod(i,16)
    js=[rr*16+cc for rr in range(max(0,r-1),min(16,r+2))
        for cc in range(max(0,c-1),min(16,c+2))
        if rr*16+cc not in rows]
    neighbor[i]=js
    excess[i]=P0[i]-P0[js].mean(0)
Ph=P0-0.5*excess
save_edit(pn,Ph)
cand=W/"candidate_register_position_half";cand.mkdir(exist_ok=True)
torch.save(Ph,cand/(pn+".pt"))
torch.save({"rows":rows,"neighbors":neighbor,"excess":excess},
           W/"subspaces"/"register_position_excess.pt")
(W/"current_candidate.json").write_text(json.dumps(
    {"name":"register_position_half","rows":rows,"alpha":0.5}))
print("Rows and norms",[(i,round(float(norms[i]),1),round(float(Ph[i].norm()),1)) for i in rows])
print("Unchanged normal rows",torch.equal(P0[[i for i in range(256) if i not in rows]],
                                         Ph[[i for i in range(256) if i not in rows]]))
print("Relative full tensor change",round(float((Ph-P0).norm()/P0.norm()),4))
with (W/"notes.md").open("a") as f:
    f.write("\n## Final new candidate family: extreme visual positional rows\n"
            "Five SigLIP positional rows13,47,48,193,243 have norms94-314 versus median18.2. "
            "They attract substantial L14H0 attention. Earlier low-frequency-position edit protected these rows; "
            "downstream register-head attenuation tested a different mechanism and failed. "
            "Now test upstream allocation: P_i'=P_i-.5(P_i-mean(neighboring ordinary rows)) for just these5rows. "
            "Every ordinary position is unchanged. No claim that high norm alone establishes harm; causal test required. "
            "Reserve remaining passes for dose testing and independent validation, not additional candidate fishing.\n")
```
