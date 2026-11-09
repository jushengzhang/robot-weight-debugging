Weakening the selected MLP channels increased both loop failures and mean gripper flips, while reducing success. That suggests they stabilize gripper switching rather than cause the loop. I will test the opposite, preselected dose—1.5 times their original output—to see whether stronger stabilization helps.
```python
def apply_mlp_channel_gain(layer, gain, count=32):
    for n in list(edited()):
        drop_edit(n)
    ix=mlp_candidates[layer]["indices"][:count]
    n=E+f"layers.{layer}.mlp.down_proj.weight"
    w=load_original(n).clone()
    w[:,ix]*=gain
    save_edit(n,w)
    spec={"name":f"E{layer}_MLP_grip{count}_gain{gain}",
          "layer":layer,"gain":gain,"indices":ix.tolist()}
    (W/"current_candidate.json").write_text(json.dumps(spec,indent=2))

apply_mlp_channel_gain(16,1.5)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 24: E16 MLP grip32 gain 0.5 rejected\n"
            "- Paired screen 32/64 -> 27/64, -7.8 pp (SE 4.0), 1 rescue / 6 losses; loops 20 -> 23; mean flips 12.55 -> 14.56.\n"
            "- Original result drifted by one success from earlier repeats (31 -> 32); always use the within-pass paired baseline.\n"
            "- Attenuation destabilizes, contrary to an excessive-feedback account. Next fixed opposite dose: same 32 columns at gain 1.5.\n")
print("Active candidate:",(W/"current_candidate.json").read_text())
```
