I’ll test a route the earlier work did not fully ablate: text entering the video observation encoder. The first intervention removes half of its text-specific input component while exactly preserving the proprioception span. This is a causal test, not yet a proposed final repair.```python
res4 = json.loads((W / "probe_4" / "results.json").read_text())
print("Probe result schema:", str(res4)[:2200])
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
def text_route_attenuation(expert, dose):
    name = f"mot.mixtures.{expert}.text_embedding.0.weight"
    t = load_original(name)
    keep = (t @ Qprop) @ Qprop.T
    e = t - dose * (t - keep)
    save_edit(name, e)
    print(expert, "dose", dose, "relative change", float((e-t).norm()/t.norm()),
          "proprio relative error", float(((e-t)@prop).norm()/(t@prop).norm()))
clear_edits()
text_route_attenuation("video", 0.5)
with (W / "notes.md").open("a") as f:
    f.write("\nPass4: 72-input fixed-noise original action/residual reference saved.\n"
            "Current causal diagnostic: video text_embedding.0.weight right-projected halfway "
            "toward the exact9D proprio span. Action expert unchanged; not a final repair.\n")
```
