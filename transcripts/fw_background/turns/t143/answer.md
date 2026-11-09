Object-layout outcomes matched at 89/128. I will save the audit and prepare the robot-initial-state variants; sensor noise will be the final category.
```python
print(json.dumps(summarize_rollback(38),indent=1))
print("\nNext request:")
print(json.dumps({"type":"rollout","model":"both",
                  "episodes":audit_manifests["Robot Initial States"]},separators=(",",":")))
```
