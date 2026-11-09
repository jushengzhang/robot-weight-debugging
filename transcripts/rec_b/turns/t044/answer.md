Removing the head produced no net success gain: two rescues and two regressions. I will reject it and broaden the diagnosis to the late expert residual and MLP pathways, using more saved observations rather than relying on a few attention maps.
```python
for n in list(edited()):
    drop_edit(n)
with (W/"notes.md").open("a") as f:
    f.write("\n## Rejected complete L16h3 ablation, pass17\n"
            "O16 head3 zeroed:26/48->26/48,2 rescues/2 regressions,loops21->19. "
            "Attention fixation alone has not localized a repair. Reset. "
            "Next analyze late expert residual/MLP and action trajectories on more observations.\n")
print("Probe metadata:")
for k in [2,7,11,13,14]:
    p=W/f"probe_{k}"
    js=list(p.glob("*.json"))
    for q in js:
        print(k,q.name,q.read_text()[:2700])
print("\nProbe shapes:")
for q in sorted((W/"probe_7").glob("*.npy"))[:10]:
    a=np.load(q)
    print(q.name,a.shape,a.dtype)
a=np.load(W/"roll_1_acts"/"s1274_close_seed17_original.npz")
print("Roll activation shapes:", {k:a[k].shape for k in a.files})
print("Example frames:", [p.name for p in sorted((W/"roll_1_frames"/"s1274_close_seed17_original").iterdir())][:12])
print("Partition preview:", list(parts["partition"].items())[:5])
print("Weight names at late expert:", [n for n in NAMES if "gemma_expert.model.layers.16." in n])
```
