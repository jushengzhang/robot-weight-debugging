I will check the saved attention maps and vision diagnostics to distinguish useful global “register” patches from harmful fixed-location cues. Large positional embeddings alone do not justify an edit; I want to know whether the instruction-sensitive visual heads actually rely on those patches.
```python
for name in ["probe_6", "probe_20", "probe_21", "subspaces", "fwd"]:
    p = W/name
    print("\n", name)
    print([x.name for x in sorted(p.iterdir())][:25])
    for fn in ["index.json", "summary.json"]:
        if (p/fn).exists():
            data=json.loads((p/fn).read_text())
            print(fn, str(data)[:2400])
print("\nSaved input list:", (W/"semantic_swap_inputs.json").read_text()[:2200])
print("\nFirst baseline record:", (W/"roll_1.jsonl").read_text().splitlines()[0][:3200])
print("\nSplit manifest:", (W/"split_manifest.json").read_text()[:1500])
print("\nRelevant embedding/vision names:")
for n in NAMES:
    if ("position_embedding" in n or "post_layernorm" in n or
        "multi_modal_projector" in n or "patch_embedding" in n):
        print(n, SHAPES[n])
```
