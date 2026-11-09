Camera-viewpoint outcomes matched at 47/128, again with no paired success difference. I will save that rollback result and prepare the reserved language-instruction variants next.
```python
print(json.dumps(summarize_rollback(36),indent=1))
assert list(edited()) == [noop_name]
assert torch.equal(load(noop_name), load_original(noop_name))
print("\nNext request:")
print(json.dumps({"type":"rollout","model":"both",
                  "episodes":audit_manifests["Language Instructions"]},separators=(",",":")))
```
