I will check the transition timing and pose changes in the state-discrimination panel. The desired pattern is earlier release or a changed approach for empty grasps, with held-object commands preserved; any indiscriminate opening would rule out this edit.
```python
p14=json.loads((W/"probe_14/results.json").read_text())
for ii,z in enumerate(grip_probe_meta):
    dd={p["arm"]:p for p in p14 if p["idx"]==ii}
    ao=np.load(dd["orig"]["action"]); ae=np.load(dd["edit"]["action"])
    print(ii,z["id"],"empty-loop" if ii<8 else "control",
          "g","".join("+" if v>0 else "-" for v in ao[:,6]),"->","".join("+" if v>0 else "-" for v in ae[:,6]),
          "XYZ",round(float(np.sqrt(np.mean((ae[:,:3]-ao[:,:3])**2))),4),
          "grip_RMS",round(float(np.sqrt(np.mean((ae[:,6]-ao[:,6])**2))),4))
```
