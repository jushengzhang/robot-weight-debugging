The probe tool requires the original registered rollout paths and rejected the unchanged copies without charging a pass. I’ll use those registered paths directly.
```python
registered_paths = [p["path"] for p in pitems]
print(json.dumps(registered_paths, separators=(",",":")))
```
