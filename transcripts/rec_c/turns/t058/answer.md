Layer 17 has four almost entirely self-attending heads, while heads 1 and 7 retain substantial prefix attention. Broadly weakening the self-attending heads previously hurt performance. I will instead double only heads 1 and 7, testing whether stronger late observation conditioning helps while leaving the main self-attention paths intact.
```python
name="paligemma_with_expert.gemma_expert.model.layers.17.self_attn.o_proj.weight"
weight=load_original(name)
changed=weight.clone()
for h in [1,7]:
    changed[:,h*256:(h+1)*256]*=2.0
save_edit(name,changed)
print("edited heads L17 H1,H7 factor2; unchanged fraction",
      round(float((weight==changed).float().mean()),3))
print("backup",backup_edits("candidate_L17_prefix_heads_x2"))
with (W/"notes.md").open("a") as f:
    f.write("\n## Next intervention: strengthen late prefix-reading heads\n"
            "L17 H0/H3/H4/H6 are ~99% action-self; broad attenuation hurt earlier. "
            "L17 H1/H7 retain prefix evidence (H1 ~55-77%, H7 ~27-80% on4 diagnostics), "
            "unlike the other heads. Double only their o_proj column blocks, preserving self-head denoising. "
            "Mechanism hypothesis: late evidence integration underweighted relative to action-state recycling. "
            "Requires paired rollout, dose response, and fresh-scene validation.\n")
del weight,changed
```
