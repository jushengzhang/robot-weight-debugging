Sharpening E8H7 did not help: success fell from 31/64 to 29/64, with one rescue and three losses. I will revert it. Most earlier interventions changed how the action expert uses observations; I will now inspect the saved visual-language representations for a more upstream source of poor localization.
```python
drop_edit(qname)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 39: E8H7 query sharpening rejected\n"
            "- Paired screen: 31/64 -> 29/64, -3.1 pp (SE 3.1), 1 rescue / 3 losses; loops 21 -> 22. Reverted.\n"
            "- The intended entropy reduction was causal, but not a recovery repair. Current edits none.\n")
for k in [4,15,20,26,31]:
    p=W/f"probe_{k}"
    rpath=p/"results.json"
    if not rpath.exists(): continue
    rr=json.loads(rpath.read_text())
    print("\nProbe",k,"records",len(rr),"keys",list(rr[0]))
    if "activations" in rr[0]:
        z=np.load(rr[0]["activations"])
        print("activation keys/shapes",[(key,z[key].shape) for key in z.files][:45])
    else:
        fs=list(p.glob("*acts.npz"))
        if fs:
            z=np.load(fs[0])
            print("activation keys/shapes",[(key,z[key].shape) for key in z.files][:45])
print("\nWeight name examples (vision attention):",
      [n for n in NAMES if "vision_model.encoder.layers.10" in n][:16])
print("Catalog top-level:",type(cat),len(cat))
print("Catalog first item:", str(cat[0] if isinstance(cat,list) else list(cat.items())[:1])[:2500])
```
