L15 H5 is the earliest recorded head with essentially no direct observation attention, just before the large late-layer growth in the gripper-axis signal. I will halve only this head’s output block to test whether its action-to-action contribution reinforces the repeated grasp sequence.
```python
def apply_head_gain(layer, head, gain):
    for n in list(edited()):
        drop_edit(n)
    name=E+f"layers.{layer}.self_attn.o_proj.weight"
    w=load_original(name)
    w[:,head*256:(head+1)*256] *= gain
    save_edit(name,w)
    spec={"name":f"E{layer}H{head}_O_{gain}","layer":layer,"head":head,"gain":gain}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))

apply_head_gain(15,5,0.5)
print("Active:",(W/"current_candidate.json").read_text())
with (W/"notes.md").open("a") as f:
    f.write("\n## Self-conditioning hypothesis\n"
            "- Saved maps: E15H5 ~100% action-token attention in all four matched observations; E16H0 ~98.6%, E16H1 ~99.8%, E16H5 ~99.6%; several E17 heads also ~100%.\n"
            "- Grip-axis residual std in 48 t0 probes grows L14 12.8 -> L15 27.1 -> L16 50.0 -> L17 71.9 at first denoising call.\n"
            "- Candidate E15H5_O_0.5: earliest isolated self-only head before this amplification. This is a causal test, not a claim that self-attention is inherently bad.\n")
```
