The expected H15 edit is installed. I will verify that it matches the frozen checkpoint and recover the exact pre-failure panel, so its preservation audit uses the same scenes, gripper starts, and seeds as I065.
```python
import hashlib

def file_digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:20]

for folder in ("best_H15", "edited"):
    files = [p for p in (W / folder).rglob("*") if p.is_file()]
    print(folder, [(str(p.relative_to(W / folder)), file_digest(p)) for p in files])

prefail = json.loads((W / "prefail_preservation_episodes.json").read_text())
print("Preservation panel:")
print(json.dumps(prefail, separators=(",", ":")))

r47 = [json.loads(line) for line in (W / "roll_47.jsonl").read_text().splitlines() if line.strip()]
print("Record count:", len(r47))
print("First-record keys:", sorted(r47[0]))
print("First-record scalar fields:", {
    k: v for k, v in r47[0].items()
    if isinstance(v, (str, int, float, bool)) or v is None
})
```