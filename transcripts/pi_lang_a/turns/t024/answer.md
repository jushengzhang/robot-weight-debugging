The save function always writes float32 files, so that attempt cannot reduce transfer memory. I will restore the exact sparse-row edit and verify that every other row is unchanged. I will then use a small paired rollout for the ablation, avoiding the failing probe route.

```python
print("Original embedding already bf16-exact:",torch.equal(E0,E0.to(torch.bfloat16).float()))
apply_aliases({"basin":"bowl"},1.0,"basin_only")
saved_E=load(emb_name)
row_changed=(saved_E!=E0).any(dim=1)
print("Changed row IDs:",torch.where(row_changed)[0].tolist())
print("Protected-token changes:",int(row_changed[list(canonical_ids)].sum()))
del saved_E,row_changed
print("Current edit: exact basin-only substitution.")
```
