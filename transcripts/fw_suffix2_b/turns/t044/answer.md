The saved repair removes selected tagged-versus-clean context directions while preserving a 521-dimensional span of clean, paraphrased, and proprioceptive features. I will confirm the probe file layout before testing it; this protection is exact only on the saved span, so rollout safety checks will still be necessary.
```python
print("WORK", str(W))
print("PROBE FILES", sorted(p.name for p in (W/"probe_4").iterdir())[:15])
for p in (W/"probe_4").iterdir():
    if p.suffix == ".json":
        print(p.name, p.read_text()[:2800])
print("REPLAY", sorted(p.name for p in (W/"replay").iterdir())[:6])
print("SPLIT SAMPLE", sp["diagnostic"][:1], sp["development"][:1])
print("QUERY SPECS", meta["query_specs"])
print("PROJECTION CHECK", float((proj["U"][:,:64].T @ proj["B_nearest"][:,:64] - torch.eye(64)).abs().max()))
print("PROBE 4 ARRAY SAMPLE")
for p in sorted((W/"probe_4").glob("*.npy"))[:4]:
    a = np.load(p, mmap_mode="r")
    print(p.name, a.shape)
print("CANONICAL INPUTS", [str(Path("/work/replay")/Path(p).name) for p in paths][:3])
```
