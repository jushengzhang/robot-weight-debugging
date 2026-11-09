Robot-initial-state outcomes matched at 100/128. I will record that result and prepare the final sensor-noise audit; the submitted checkpoint will remain unchanged.
```python
print(json.dumps(summarize_rollback(39),indent=1))
assert list(edited()) == [noop_name]
assert torch.equal(load(noop_name), load_original(noop_name))
print("\nFinal forward request:")
print(json.dumps({"type":"rollout","model":"both",
                  "episodes":audit_manifests["Sensor Noise"]},separators=(",",":")))
```
