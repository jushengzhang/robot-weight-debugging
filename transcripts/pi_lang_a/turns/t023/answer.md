The service still cannot allocate the full-precision embedding override. I will check how the edit is stored and try saving it in the model’s bf16 precision, if the provided save function supports that. This changes no intended weight values beyond the casting that inference already performs.

```python
edit_files=list((W/"edited").glob("**/*"))
print("edit files:",[(p.name,p.stat().st_size) for p in edit_files if p.is_file()][:12])
print("canonical instructions containing 'compartment':",[s for s in canonical_texts if "compartment" in s])
# Store the same basin-only intervention in bf16, matching inference precision.
E_bf=E0.to(torch.bfloat16)
E_bf[word_ids("basin")[0]]=E0[word_ids("bowl")[0]].to(torch.bfloat16)
save_edit(emb_name,E_bf)
del E_bf
print("after save:",[(p.name,p.stat().st_size) for p in (W/"edited").glob("**/*") if p.is_file()][:12])
with (W/"notes.md").open("a") as f:
    f.write("\nSecond basin-only probe attempt also OOM (not counted). "
            "Trying bf16 storage through save_edit to reduce override-transfer memory, "
            "with identical intended inference values. "
            "Compartment is protected: literal long-suite instructions use it, so no edit to that row.\n")
```
