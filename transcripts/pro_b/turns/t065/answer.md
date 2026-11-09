The controlled swap strongly changes actions, so the model is not simply ignoring language. Its effect also spreads into the VLM’s image-token representations, and the expert’s response jumps sharply at layer 14. This suggests that instruction-conditioned visual features—not direct attention to words—are the main control route. I will examine layer 14 and neighboring heads under the same semantic swap.
```python
emb=load_original(embname)
for ia,ib in swap_ids:
    row=emb[ia].clone()
    emb[ia]=emb[ib]
    emb[ib]=row
save_edit(embname,emb)
del emb
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 20: controlled noun swap reveals the main language route\n"
            "Same images/noise, tomato-sauce<->salad-dressing embedding swap "
            "changes initial XYZ chunks by RMS .142/.125 (task/semantic). "
            "VLM image representations also change (base relative delta about.13 "
            "atL12 and.16 atL15), proving instruction-conditioned visual processing. "
            "Expert first-denoise relative difference jumps L13 .0805 ->L14 .1878 "
            "(task) and .0878->.2086 (semantic). "
            "Direct word-reading heads previously tested are not the principal route. "
            "Embeddings restored after probe20, temporary swap reinstated for attention localization.\n")
print("Attention probe inputs:",[swap_inputs[i] for i in [0,1,5,6]])
print("Layer14 parameter norms:")
for part in ["self_attn.o_proj.weight","mlp.down_proj.weight",
             "input_layernorm.dense.bias","post_attention_layernorm.dense.bias"]:
    a=load_original(f"{ex}.layers.14.{part}")
    print(part,tuple(a.shape),round(float(a.norm()),3))
```
