Full strength yields ten rescues and one regression. Before choosing it, I will identify the regression and test whether the benefit comes from task-specific embedding directions or merely from changing the large common component shared by token embeddings. This control is important because a successful alias edit alone does not identify which part of the row matters.

```python
print("pass12 valid summary:",paired_summary(12))
for key,a,b in paired_records(12):
    if a["success"]!=b["success"]:
        print("RESCUE" if b["success"] else "REGRESSION",key,a["instruction"])
print("Dairy contexts:")
for r,toks in zip(diag_records,diag_token_sets):
    if word_ids("dairy")[0] in toks:
        print(r["suite"],r["id"],r["success"],r["instruction"])
shutil.copytree(W/"edited",W/"candidate_full_weights",dirs_exist_ok=True)
common_u=Emean/Emean.norm()
def apply_component_control(meta, component, tag):
    E=E0.clone()
    rows=[]
    for m in meta:
        si,ti,a=m["source_id"],m["target_id"],m["alpha"]
        delta=a*(E0[ti]-E0[si])
        common=torch.dot(delta,common_u)*common_u
        change=common if component=="common" else delta-common
        E[si]=E0[si]+change
        rows.append((m["source"],round(float(common.norm()/delta.norm()),3)))
    save_edit(emb_name,E)
    del E
    (W/f"{tag}.json").write_text(json.dumps({"component":component,"rows":meta},indent=2))
    return rows
common_fractions=apply_component_control(candidate_full_meta,"common","common_direction_control")
print("Fraction of edit norm along global embedding mean:",common_fractions)
with (W/"notes.md").open("a") as f:
    f.write("\nPass12 full17 candidate:48 requested,5 missing; "
            f"valid {paired_summary(12)}. Saved candidate_full_weights. "
            "Mechanistic control: project each proposed row delta onto the global embedding-mean direction, "
            "leaving all orthogonal token information unchanged. No fitting/optimization.\n")
```
