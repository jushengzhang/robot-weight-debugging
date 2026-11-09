Language outcomes matched at 77/128 with no paired disagreements. I will record the trajectory check and prepare the object-layout variants next.
```python
print(json.dumps(summarize_rollback(37),indent=1))
print("\nNext request:")
print(json.dumps({"type":"rollout","model":"both",
                  "episodes":audit_manifests["Objects Layout"]},separators=(",",":")))
```
