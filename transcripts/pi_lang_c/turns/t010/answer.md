I will summarize attention by head and inspect the probe layout. The key distinction is between too little language attention and strong attention to misleading lexical features; those mechanisms call for very different repairs.
```python
def read_probe(k):
    return json.loads((W / f"probe_{k}/results.json").read_text())
p3=read_probe(3); p4=read_probe(4)
print("p3 type:", type(p3).__name__, "example:", str(p3)[:3000])
print("p4 type:", type(p4).__name__, "example:", str(p4)[:1800])
def entries(p):
    if isinstance(p,list): return p
    for k in ["results","inputs","records"]:
        if k in p and isinstance(p[k], list): return p[k]
    return list(p.values())
e3=entries(p3); e4=entries(p4)
print("Entry keys:", list(e3[0]))
print("Files probe 4:", [p.name for p in (W / "probe_4").iterdir()][:10])
```
