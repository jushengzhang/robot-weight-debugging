The edit-size check returned zero, which suggests `load_original` returned a shared cached tensor that I modified in place. I will restore that in-memory reference and re-save the intended edit from an explicit clone before running inference, so the original comparison remains well defined.
```python
# Undo only the known in-memory mutation, then work exclusively on clones.
a[:,6] *= 2.0
orig_in = load_original("action_in_proj.weight").clone()
edit_in = orig_in.clone()
edit_in[:,6] *= 0.5
save_edit("action_in_proj.weight", edit_in)
print("original grip-column norm", round(float(orig_in[:,6].norm()),6))
print("edited grip-column norm", round(float(load("action_in_proj.weight")[:,6].norm()),6))
print("relative tensor change", round(float((edit_in-orig_in).norm()/orig_in.norm()),6))
print("other columns unchanged", bool(torch.equal(edit_in[:,:6],orig_in[:,:6])
                                     and torch.equal(edit_in[:,7:],orig_in[:,7:])))
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,candidate/p.name)
```
