The earlier results suggest that masking suffix-like tokens alone did not recover clean behavior, and aggressive masking sometimes made it worse. I’ll finish reading the validation results and check how text enters each expert so the next tests distinguish direct suffix effects from changes to the task-token embeddings.```python
print(prior[-6500:])
print("\n--- ARCHITECTURE NAMES (first block only) ---")
print("Total tensors:", len(NAMES), "edits:", len(edited()))
for name in NAMES:
    if ".blocks.0." in name or (not re.search(r"(blocks|layers)\.\d+\.", name)):
        print(name, SHAPES[name])
print("\nWORK entries:", [p.name for p in W.iterdir()])
```
