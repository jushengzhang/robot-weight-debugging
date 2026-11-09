The contact-specific wrist direction has the expected sign in all eight reserved scenes, even after removing the aperture component; the base-camera contrast is much less consistent. I will double this one direction in the visual projector, preserving orthogonal features and the layer-normalization offset, to test whether clearer contact evidence changes the policy’s grasp decisions.
```python
proj=f"{'paligemma_with_expert.paligemma.model'}.multi_modal_projector.linear"
pw=load_original(proj+".weight").clone()
pb=load_original(proj+".bias").clone()
u=torch.load(W/"visual_contact_direction.pt",weights_only=False)["direction"].float()
u=u/u.norm()
pu=pw@u
gain=2.0
neww=pw+(gain-1)*torch.outer(pu,u)
newb=pb-(gain-1)*pu*torch.dot(u,lnb)
save_edit(proj+".weight",neww)
save_edit(proj+".bias",newb)
print("contact/aperture cos",round(float(u@ap),7))
print("projector contact sensitivity / isotropic reference",
      round(float(pu.norm()/(pw.norm()/math.sqrt(pw.shape[1]))),3))
print("weight relative change",round(float((neww-pw).norm()/pw.norm()),5),
      "bias change norm",round(float((newb-pb).norm()),4))
print("bf16 direction gain",round(float(((neww.bfloat16().float()-pw.bfloat16().float())@u).norm()/pu.norm()+1),4))
cd=W/"candidate_visual_contact_x2"; cd.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,cd/p.name)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass32: contact-specific visual direction\n"
            "Stable-close observations require preceding10 executed close commands. Within22 scenes, "
            "contrast width0.018-0.068m (contact) vs<0.008m (empty closure). "
            "14 source scenes;8 independently reserved for feature check. SigLIP26 wrist mean-unit contrast "
            "is centered, LN-scaled, then orthogonalized against earlier aperture direction. "
            "Reserved pair cosines all8 positive, mean0.199 (source0.332); base-camera contrast only5/8. "
            "Pose changes remain a confound, so call this contact-associated, not a proven object-held detector.\n"
            "Candidate doubles only this projector input direction with LN-bias offset preservation; "
            "orthogonal directions and raw aperture axis unchanged. No success labels used in constructing direction.\n")
```
