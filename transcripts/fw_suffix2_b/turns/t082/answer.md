The earlier notes confirm that broad ablations did not recover clean behavior, while suffix gating helped only slightly. I’ll now check the current candidate’s exact construction and its latest safety results.
```python
ptext = (W / "prior_notes.md").read_text()
start = ptext.find("### Pass 23:")
print(ptext[start:][-6500:])
print("\nWORK FILES")
for p in sorted(W.iterdir()):
    if p.is_file():
        print(p.name, p.stat().st_size)
    elif not p.name.startswith(("roll_", "probe_")):
        print(p.name + "/")
print("\nCURRENT EDITS:", len(edited()), list(edited())[:8])
for k in (28, 29):
    p = W / f"roll_{k}.jsonl"
    if p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        print("\nROLLOUT", k, "rows", len(rows))
        print("first:", json.dumps(rows[0])[:1300])
        groups = {}
        for r in rows:
            m = str(r.get("model", "unknown"))
            g = groups.setdefault(m, [0, 0])
            g[0] += int(bool(r.get("success", False)))
            g[1] += 1
        print("counts:", groups)
```
