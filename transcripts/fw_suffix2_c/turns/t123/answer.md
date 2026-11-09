The prior work found that stronger suffix gating could make actions worse, and its early rollout gains were inconclusive. I’ll finish that history and verify the active files against the frozen copy before using the remaining validation passes.
```python
print(prior_text[16500:])

manifest = json.loads((root / "frozen_manifest.json").read_text())
def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

assert set(edited()) == set(manifest), "Active tensor names differ from frozen names"
checks = []
for name in sorted(manifest):
    active = root / "edited" / (name + ".npy")
    frozen = root / "frozen_video_value_half" / (name + ".npy")
    assert active.exists() and frozen.exists(), name
    checks.append((sha256_file(active), sha256_file(frozen), manifest[name]))
print("\nFROZEN_FILE_EQUALITY", all(a == b for a, b, _ in checks))
print("MANIFEST_FILE_HASH_MATCH", all(a == c for a, _, c in checks))
print("HELPER_SCRIPTS", sorted(p.name for p in root.glob("*.py")))
print("JSON_ARTIFACTS", sorted(p.name for p in root.glob("*.json")))
```
