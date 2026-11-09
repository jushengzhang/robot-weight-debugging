The open-control records use a different label field. I will handle both formats and finish checking the input links before probing.
```python
def group_key(r):
    return r.get("split", "control") + "_" + r.get("class", r.get("role", "unknown"))
print(Counter(group_key(r) for r in m))
for i in [0, 27, 28, 43, 44, 59]:
    p = Path(m[i]["input"])
    source = Path(m[i].get("source", m[i]["input"]))
    print(i, group_key(m[i]), "symlink", p.is_symlink(),
          "resolves_to_source", p.resolve() == source.resolve(), "exists", p.exists())
print("Forward-compatible alias prefix:", str(W/"value_contact_probe"))
```
