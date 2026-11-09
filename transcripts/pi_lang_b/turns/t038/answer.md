Half-strength destination anchoring yields 12 repairs and one regression, a substantially larger diagnostic gain. I will inspect the changed tasks and the reported error, then test full-strength destination anchoring as the planned second dose.
```python
r15=read_roll(15)
print("destination half",paired_summary(r15,True))
for r in r15:
    if r.get("termination")=="error":
        print("error record",json.dumps(r,ensure_ascii=True)[:1200])
shutil.copytree(W/"edited",W/"checkpoints"/"lex_dest_half",dirs_exist_ok=True)
spec_dest_full={**spec_base,**{s:(t,1.) for s,t in pairs_dest.items()}}
apply_spec(spec_dest_full,"lex_dest_full")
with (W/"notes.md").open("a") as f:
    f.write("\nPass15: 20-row candidate (rounded/mixing reduced to .5; unit->cabinet, furniture->cabinet, "
            "structure->rack at .5): +17.19 pp SE5.2, 12 repairs/1 regression on dev goal64; "
            "9 matched nulls, 1 edited error counted as failure. Snapshot lex_dest_half. "
            "Testing full destination dose before candidate freeze.\n")
```
