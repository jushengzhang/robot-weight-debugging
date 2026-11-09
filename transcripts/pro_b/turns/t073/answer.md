The saved inputs use more than one observation-dictionary layout. I will inspect those layouts and handle both explicitly before calculating color statistics; no weight edit is active.
```python
for r in r1[:5]:
    p=Path(r["frames_dir"])/"t0000_inputs.pt"
    z=torch.load(p,map_location="cpu",weights_only=False)
    print(r["id"],"top",list(z),"element",list(z.get("element",{})))
print("Active edits:",edited())
```
