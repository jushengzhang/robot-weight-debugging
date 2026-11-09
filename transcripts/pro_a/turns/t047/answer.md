Halving the projector mode harmed environment performance and did not improve task or position success, so I will reject it too. I will now inspect the late VLM/expert interface: visual residual norms grow sharply there, and we need to know whether the action expert reads useful object information or mostly fixed summary tokens.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass17: projector top mode retain50%\n"
        "Dev69->64/128, -3.9 +/-2.6pp, 3 wins/8 losses. "
        "Environment17->11, object23->21, position3 unchanged, semantic21->24, task5 unchanged. "
        "REJECT: common projector carrier is important; removing more does not repair target failures.\n"
    )
for n in list(edited()):
    drop_edit(n)

# Weight-only inspection: late VLM residual-producing projections and RMSNorms.
vlm_root = "paligemma_with_expert.paligemma.model.language_model.layers"
print("VLM layer name examples:", [n for n in NAMES if "language_model.layers.15." in n][:12])
for l in [3, 6, 9, 12, 13, 14, 15, 16, 17]:
    found = [n for n in NAMES if f"language_model.layers.{l}." in n]
    desc = []
    for suffix in ["mlp.down_proj.weight", "self_attn.o_proj.weight",
                   "input_layernorm.weight", "post_attention_layernorm.weight"]:
        n = next((n for n in found if n.endswith(suffix)), None)
        if n:
            t = load_original(n)
            if t.ndim == 2:
                rn = t.norm(dim=1)
                desc.append((suffix.split(".")[0]+"."+suffix.split(".")[1],
                             round(float(t.norm()), 1), round(float(rn.max()/rn.median()), 2),
                             rn.topk(3).indices.tolist()))
            else:
                desc.append((suffix.split(".")[0], round(float(t.abs().median()), 3),
                             round(float(t.abs().max()), 3), t.abs().topk(3).indices.tolist()))
    print(l, desc)

# Record exact untouched replay inputs for late attention inspection.
late_inputs = [
    str(W/"roll_1_frames"/"e1946_seed7_original"/"t0000_inputs.pt"),
    str(W/"roll_1_frames"/"e1923_seed7_original"/"t0000_inputs.pt"),
    str(W/"roll_1_frames"/"e1896_seed7_original"/"t0100_inputs.pt"),
    str(W/"roll_1_frames"/"e1490_seed7_original"/"t0100_inputs.pt"),
]
print("Late probe paths exist:", [(p, Path(p).exists()) for p in late_inputs])
(W/"late_probe_inputs.json").write_text(json.dumps(late_inputs))
```
