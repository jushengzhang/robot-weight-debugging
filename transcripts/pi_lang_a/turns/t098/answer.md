The canonical replay has finished. I will compare its saved action and activation arrays directly before interpreting it as an invariance result.
```python
probe46 = json.loads((W / "probe_46" / "results.json").read_text())
def compact_schema(x, depth=0):
    if isinstance(x, dict):
        return {k: compact_schema(v, depth+1) if depth < 2 else
                (f"{type(v).__name__}[{len(v)}]" if isinstance(v, (dict, list)) else str(v)[:150])
                for k, v in x.items()}
    if isinstance(x, list):
        return {"length": len(x), "first": compact_schema(x[0], depth+1) if x else None}
    return str(x)[:180]
print(json.dumps(compact_schema(probe46), indent=1)[:6800])
print("Probe files, first 12:", [p.name for p in sorted((W / "probe_46").iterdir())[:12]])
```
