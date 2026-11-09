I’ll measure the combined projection first. I’ll also inspect the temporary readout setup: if the projection still leaves a large gap, the remaining effect may lie in individual task-token changes that a uniform context average misses.
```python
m16=compare_screen(16)
print("=== INSTRUMENTATION TENSORS ===")
dd=W/"fwd/probe_10_weights"
for p in dd.iterdir():
    a=np.load(p,mmap_mode="r")
    print(p.name, a.shape)
print("=== ACTION BLOCK NAMES ===")
for n in NAMES:
    if n.startswith("mot.mixtures.action.blocks.0."):
        print(n,SHAPES[n])
print("=== CONTEXT FEATURE SUMMARY ===")
fd=torch.load(W/"video_context_features.pt",weights_only=False)
H=fd["features"]
for j in range(3):
    print("readout",j,"mean_norm",float(H[:,j].norm(dim=1).mean()),
          "clean_tag_diff_norm",float((H[:48,j]-H[48:96,j]).norm(dim=1).mean()),
          "clean_vs_masked",float((H[48:96,j]-H[48:96,0]).norm(dim=1).mean()))
```
