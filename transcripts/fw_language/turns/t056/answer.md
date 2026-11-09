The tool requires individual registered input files and rejected the convenience directory without charging a pass. I will use those original file paths directly and reduce the bank to the ten diagnostic pairs, canonical controls, and one successful representative for each remaining task.
```python
# Remove the unsupported convenience directory; no source artifacts are changed.
for p in bank.iterdir():
    if p.is_symlink(): p.unlink()
bank.rmdir()

by_path={r["input"]:r for r in bank_records}
selected=[]
selected_paths=set()
def add_record(r):
    if r["input"] not in selected_paths:
        selected.append(r); selected_paths.add(r["input"])
for p in pr4:
    add_record(by_path[str(Path(p["input"]).resolve())])
for r in bank_records:
    if r["category"]!="Language Instructions": add_record(r)
covered={(r["suite"],r["task"]) for r in selected if r["success"]}
for task in sorted(set((r["suite"],r["task"]) for r in bank_records)):
    if task in covered: continue
    options=[r for r in bank_records if (r["suite"],r["task"])==task and r["success"]]
    if options:
        add_record(min(options,key=lambda r:len(r["instruction"])))
        covered.add(task)
(W/"compact_probe_manifest.json").write_text(json.dumps(selected,indent=2))
print("Inputs:",len(selected),"successful task coverage:",len(covered))
print(json.dumps([r["input"] for r in selected]))
```
