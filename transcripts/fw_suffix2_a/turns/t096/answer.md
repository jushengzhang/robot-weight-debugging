The replay folder contains all 112 diagnostic inputs, while the action reference uses a specific 56-input subset. I’ll copy that subset unchanged into a numbered folder so the next probe matches the reference ordering exactly.
```python
import shutil
replay56 = W/"reference56"
replay56.mkdir(exist_ok=True)
for i,p in enumerate(pitems):
    assert Path(p["path"]).exists()
    shutil.copy2(p["path"], replay56/f"i{i:03d}.pt")
assert len(pitems)==56
print("Copied unchanged inputs:", len(pitems))
print("Path pattern:", str(replay56/"i000.pt"), "through i055.pt")
print("Groups:", [(i,pitems[i]["label"],pitems[i]["category"]) for i in [0,23,24,47,48,55]])
```
