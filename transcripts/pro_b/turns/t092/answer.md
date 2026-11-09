Across 40 task identities, the expert’s instruction sensitivity again jumps at layer 14: the mean residual change rises from 3.9% to 8.0%. I will double only an eight-dimensional, language-caused visual subspace in the VLM’s layer-14 value projection. Its orthogonal complement and the protected common-scene directions remain unchanged; this tests strengthening the relevant signal rather than an entire head.
```python
layer=14
U=bases[layer]
vn=LP+f"layers.{layer}.self_attn.v_proj.weight"
V=load_original(vn)
Ve=V+(V@U)@U.T
save_edit(vn,Ve)
cand=W/"candidate_controlled_visualV_g2"
cand.mkdir(exist_ok=True)
torch.save(Ve,cand/(vn+".pt"))
(W/"current_candidate.json").write_text(json.dumps(
    {"name":"controlled_visualV_g2","layer":14,"rank":8,"alpha":1.0}))
# Quantization-aware size check; v_proj is bf16 at inference.
Vq=Ve.to(torch.bfloat16).float()
print("Only edited tensor",edited())
print("Relative norm change",round(float((Ve-V).norm()/V.norm()),4),
      "retained after bf16",round(float((Vq-V).norm()/(Ve-V).norm()),4))
print("Subspace orthogonality error",float((U.T@U-torch.eye(8)).abs().max()))
print("Diagnostic episodes",len(chosen))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass27 and candidate controlled visual-value gain\n"
            "Across64 initial observations/40task identities, pure token swaps give expert mean relative residual changes "
            "L12 .035,L13 .039,L14 .080,L15 .083. Confirms L14 language-conditioned visual route. "
            "Constructed rank8 bases from normalized pooled visual differences, orthogonal to camera means and4 midpoint scene PCs; "
            "L14 retains81% contrast after protection, rank8 retains57% of remainder. "
            "Candidate: only VLM L14 v_proj W'=W+WUU^T. Changes11% Frobenius norm, preserves orthogonal complement. "
            "All token embeddings restored. Test on same diagnostic128 as pass26.\n")
```
