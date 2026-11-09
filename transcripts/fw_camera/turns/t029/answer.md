The probe tool requires the original rollout-registered file paths and rejected the replay copies without spending a pass. I will use those original paths directly; the observations and planned comparison are unchanged.```python
shutil.rmtree(replay_dir)
print(json.dumps(diag_input_paths,separators=(",",":")))
print("Registered initial observations:",len(diag_input_paths))
print("Passes spent remain:",len(ledger))
```
