The hybrid edit has only a tentative tagged gain (+4.2 points on 96 development episodes), no net paraphrase loss, and a 2.1-point clean loss. I’ll inspect how the projection protects clean language; a safer projection may be worth testing before freezing a candidate.
```python
print("CURRENT CANDIDATE", (W / "current_candidate.json").read_text())
print("HELPERS\n", (W / "analysis_helpers.py").read_text())
print("\nCANDIDATE DIRECTORIES")
for p in sorted((W / "candidates").iterdir()):
    print(p.name, "files", len(list(p.iterdir())) if p.is_dir() else p.stat().st_size)
print("\nRELEVANT TENSORS")
for n in NAMES:
    if "text_embedding" in n or "proprio" in n:
        print(n, SHAPES[n])

art = {}
for fn in ("rich_context_projection.pt", "clean_context_svd.pt",
           "masked_prefix_projection.pt", "readout_instrument_meta.pt"):
    x = torch.load(W / fn, map_location="cpu", weights_only=False)
    art[fn] = x
    print("\n", fn, type(x).__name__)
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                print(k, tuple(v.shape), str(v.dtype))
            elif isinstance(v, (list, dict)):
                print(k, str(v)[:450])
            else:
                print(k, v)
print("\nFWD ENTRY LIST", [p.name for p in (W/"fwd").iterdir()][:15])
print("\nREPLAY PATHS", (W / "replay_paths.json").read_text()[:550])
print("\nLATEST NOTES\n", (W / "notes.md").read_text()[-1800:])
```
