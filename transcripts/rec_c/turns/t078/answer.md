Neither the common-channel ablation nor the selective loop detector supports a reliable repair. One clean visual-state hypothesis remains untested: aperture information may need strengthening rather than suppression. I will double the previously measured visual-aperture direction, preserving all orthogonal directions and the LayerNorm offset.
```python
for n in list(edited()):drop_edit(n)
u=torch.load(W/"visual_grip_direction.pt",weights_only=False)["direction"].float()
u/=u.norm()
proj="paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
ww=load_original(proj+"weight");bb=load_original(proj+"bias")
lnb=load_original(next(n for n in NAMES if n.endswith("vision_model.post_layernorm.bias")))
alpha=-1.0
wu=ww@u
save_edit(proj+"weight",ww-alpha*torch.outer(wu,u))
save_edit(proj+"bias",bb+alpha*wu*(u@lnb))
print("visual aperture gain",1-alpha,"relative W change",float(torch.outer(wu,u).norm()/ww.norm()))
print("backup",backup_edits("candidate_visual_grip_x2"))
with (W/"notes.md").open("a") as f:
    f.write("\nNext causal direction test: visual aperture gain2.0 (opposite of passes17/18 suppression). "
            "Same noiseless visual contrast source and offset-preserving rank-one projector formula; "
            "all other directions unchanged. Hypothesis: insufficient visual gripper-state precision, "
            "not over-reliance; requires fresh validation if development succeeds.\n")
del ww,bb,lnb,wu
```
