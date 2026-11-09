The measurements do not show that visual changes vanish: their value-projection gain is above an isotropic reference at every layer. I will therefore test a narrower balance hypothesis, not claim a broken visual encoder. The final VLM value projection is a clean intervention point: boosting four visual-change directions there affects the expert’s last cross-attention readout without altering earlier visual processing or scaling all actions.
```python
content_layer=17
content_basis=content_bases[content_layer]
content_name=LP+f"layers.{content_layer}.self_attn.v_proj.weight"
v_orig=load_original(content_name)

def set_content_edit(alpha):
    for n in list(edited()):
        drop_edit(n)
    v_new=v_orig+alpha*(v_orig@content_basis)@content_basis.T
    save_edit(content_name,v_new)
    return v_new

v_new=set_content_edit(.5)
print("Candidate M:",content_name)
print("Rank:",content_basis.shape[1],
      "relative weight change:",round(float((v_new-v_orig).norm()/v_orig.norm()),4))

# Verify the analytic projection preserves the sampled nuisance span.
scale=1+load_original(LP+"layers.17.input_layernorm.weight")
lang=torch.tensor(np.stack([z["mean::vlm.16.resid::lang"] for z in z37]))
lang=lang/(lang.square().mean(-1,keepdim=True).sqrt()+1e-8)*scale
print("Fraction of sampled language-mean energy in edit subspace:",
      float((lang@content_basis).square().sum()/lang.square().sum()))

with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate M: final cross-attention visual innovation\n"
            "Pass37: visual-change V gains exceed isotropic at every layer, so no claim "
            "that vision is absent/erased. Construct rank4 basis from within-task base/wrist "
            "differences, orthogonal to sampled language and camera-mean span; no success-label fit. "
            "AtL17 this captures77.2% of remaining variation. "
            "Edit VLM17 V <- V + .5*(V U)U^T. Last-prefix-layer output is otherwise unused, "
            "so this changes only expert access to visual innovation, not upstream VLM routing. "
            "All action scales, norms and seven-dimensional readout remain unchanged.\n")
```
