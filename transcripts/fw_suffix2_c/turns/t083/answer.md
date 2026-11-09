I’ll inspect the extraction code and probe file format next. This will let me verify that the recovered features are correctly paired and stable across denoising steps, rather than treating an instrumentation artifact as tag contamination.
```python
print("MASKED SCRIPT\n", (W/"masked_readout.py").read_text())
print("\nANALYSIS HELPERS\n", (W/"analysis_helpers.py").read_text())
print("\nSCREEN HELPERS\n", (W/"screen_helpers.py").read_text())
print("\nSPECS")
for fn in ["masked_readout_spec.pt", "weighted_video_projector.pt", "weighted_video_features.pt", "video_context_features.pt"]:
    d = torch.load(W/fn, weights_only=False)
    print(fn, {k: (tuple(v.shape) if hasattr(v,"shape") else str(v)[:140]) for k,v in d.items()} if isinstance(d,dict) else type(d))
```
