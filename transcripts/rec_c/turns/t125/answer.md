The tool requires individual saved-input files rather than a directory. I will check that the manifest entries point to the original rollout inputs, then submit them explicitly; the candidate and test observations are unchanged.
```python
print("Probe group counts:")
from collections import Counter
print(Counter((r["split"], r["class"]) for r in m))
for i in [0, 27, 28, 43, 44, 59]:
    p = Path(m[i]["input"])
    print(i, m[i]["split"], m[i]["class"],
          "symlink", p.is_symlink(),
          "resolves_to_source", p.resolve() == Path(m[i]["source"]).resolve(),
          "exists", p.exists())
print("Forward-compatible alias prefix:", str(W/"value_contact_probe"))
```
