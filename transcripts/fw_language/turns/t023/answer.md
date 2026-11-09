I will use the remaining offline time to test whether the two experts share a simple inherited projection structure. This could explain why a video-only residual feature is read poorly by actions, but it will remain a hypothesis until we can observe the model under paired instructions.
```python
comparisons=[]
module_suffixes=[
    "text_embedding.0.weight","text_embedding.2.weight",
    "blocks.0.cross_attn.q.weight","blocks.0.cross_attn.k.weight",
    "blocks.13.cross_attn.v.weight","blocks.13.cross_attn.o.weight",
    "blocks.23.cross_attn.v.weight","blocks.23.cross_attn.o.weight",
    "blocks.24.self_attn.q.weight","blocks.24.self_attn.k.weight",
    "blocks.24.self_attn.v.weight","blocks.24.self_attn.o.weight",
    "blocks.29.cross_attn.q.weight"
]
for suffix in module_suffixes:
    a=load_original("mot.mixtures.action."+suffix)
    v=load_original("mot.mixtures.video."+suffix)
    crop=v[tuple(slice(0,s) for s in a.shape)]
    cos=float((a*crop).sum()/(a.norm()*crop.norm()))
    rel=float((a-crop).norm()/a.norm())
    comparisons.append({"suffix":suffix,"crop_cos":cos,"relative_difference":rel})
print("Action vs corresponding top-left video weights (screening only):")
for r in comparisons:
    print(r["suffix"],"cos=%.4f rel_difference=%.4f"%(r["crop_cos"],r["relative_difference"]))

print("\nCross-attention head spectra after excluding video sink coordinate 1938:")
non_sink=[]
for r in sorted(all_spectral,key=lambda r:r["relative_s1"],reverse=True)[:14]:
    if r["expert"]!="video":
        continue
    p=f"mot.mixtures.video.blocks.{r['layer']}.cross_attn."
    o=load_original(p+"o.weight")[:,r["head"]*128:(r["head"]+1)*128].clone()
    vh=load_original(p+"v.weight")[r["head"]*128:(r["head"]+1)*128]
    o[1938]=0
    s=product_singular_values(o,vh)
    non_sink.append({"layer":r["layer"],"head":r["head"],
                     "s1_original":r["s1"],"s1_without_1938":float(s[0])})
for r in non_sink:
    print("L%d H%d s1=%.2f -> %.2f"%(r["layer"],r["head"],r["s1_original"],r["s1_without_1938"]))
(W/"expert_static_comparison.json").write_text(json.dumps(comparisons))
(W/"non_sink_spectra.json").write_text(json.dumps(non_sink))
print("No active edits:",not bool(edited()))
```
