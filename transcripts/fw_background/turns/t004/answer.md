The first parser treated wall and floor suffixes as part of the task name, which produced an unbalanced sample. I will correct that before running the simulator so the diagnostic set covers all four suites and uses genuinely matched tasks.

```python
print("BG names sampled:")
for s in sorted(set(v["suite"] for v in variants)):
    vs = [v for v in bgdiag if v["suite"] == s]
    print(s, [v["name"] for v in vs[:6]])
print("Light endings:", [v["name"][-100:] for v in bycat["Light Conditions"][::40]])
print("BG endings:", [v["name"][-90:] for v in bycat["Background Textures"][::15]])
```
