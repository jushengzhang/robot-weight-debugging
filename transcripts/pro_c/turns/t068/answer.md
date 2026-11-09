On the fresh sample, BOS-key attenuation gained two environment successes but lost two object successes, for no overall improvement. I will now test the illumination edit on the same sample; its anchor was defined from diagnostic observations, not from these outcomes.
```python
for n in list(edited()):drop_edit(n)
patch_names=[n for n in NAMES if "vision_model.embeddings.patch_embedding" in n]
print("patch tensors",[(n,SHAPES[n]) for n in patch_names])
patch_w=next(n for n in patch_names if n.endswith(".weight"))
patch_b=next(n for n in patch_names if n.endswith(".bias"))
paxis=dcdef["patch_axis"].float()
anchor=float(dcdef["patch_anchor"])
v=load_original(patch_w); b=load_original(patch_b)
vf=v.reshape(v.shape[0],-1)
dcresponse=vf@paxis
alpha=0.5
save_edit(patch_w,(vf-alpha*torch.outer(dcresponse,paxis)).reshape_as(v))
save_edit(patch_b,b+alpha*dcresponse*anchor)
print("D restored axisnorm",float(paxis.norm()),"anchor",anchor,
      "relative_w",float((load(patch_w)-v).norm()/v.norm()),
      "relative_b",float((load(patch_b)-b).norm()/b.norm()))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass23: B0.5 fresh screen seed31023\n77->77/128,0pp SE1.6. Env19->21/33, object29->27/31, semantic19->19/20, position5->5/22, task5->5/22. Old-goal10->11, loops32->26. No independently demonstrated overall gain; not retained. Current D0.5 restored from saved closed-form definition.\n")
```
