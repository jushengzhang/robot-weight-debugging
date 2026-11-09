Before treating channel 1938 as language-specific, I will measure which module families write to it. If self-attention and MLPs maintain the same feature strongly, removing it wholesale would be especially risky; any eventual edit should isolate the language-sensitive write or read route.
```python
channel_sources = []
for layer in range(30):
    r = {"layer":layer}
    for label, suffix in (("text","cross_attn.o.weight"),
                          ("self","self_attn.o.weight"),("mlp","ffn.2.weight")):
        x = load_original(f"mot.mixtures.video.blocks.{layer}.{suffix}")
        rn = x.norm(dim=1)
        r[label+"_relative"] = float(rn[1938]/rn.median())
        r[label+"_norm"] = float(rn[1938])
        r[label+"_max_channel"] = int(rn.argmax())
        if label == "text":
            heads = x[1938].reshape(24,128).square().sum(1)
            r["text_top_head"] = int(heads.argmax())
            r["text_top_fraction"] = float(heads.max()/heads.sum())
    channel_sources.append(r)
print("Video residual channel 1938 write strengths (row norm / layer median):")
print("layer    text    self     mlp  top_text_head  head_fraction")
for r in channel_sources:
    print("%2d    %7.2f %7.2f %7.2f      %2d          %.3f" %
          (r["layer"],r["text_relative"],r["self_relative"],r["mlp_relative"],
           r["text_top_head"],r["text_top_fraction"]))
print("\nOther strong output channels in late video text attention:")
for layer in (13,15,20,22,23,24,27):
    x=load_original(f"mot.mixtures.video.blocks.{layer}.cross_attn.o.weight")
    rn=x.norm(dim=1)
    vals,ids=rn.topk(5)
    print(layer,[(int(i),round(float(v/rn.median()),2)) for i,v in zip(ids,vals)])
(W/"channel1938_sources.json").write_text(json.dumps(channel_sources))
with (W/"notes.md").open("a") as f:
    f.write("\n- Late video OV outlier modes largely share a coordinate: common PC "
            "has 0.99582 loading at residual channel 1938, |cos(ones)|=0.01064. "
            "This is a candidate massive-activation channel, not a proven defect. "
            "Weight-only next-layer reader screening highlights action L24 H17 "
            "(1.65x random-direction value sensitivity), L16 H14 and L14 H3. "
            "Must isolate language-sensitive routes rather than ablate the channel.\n")
print("No edits:",not bool(edited()))
```
