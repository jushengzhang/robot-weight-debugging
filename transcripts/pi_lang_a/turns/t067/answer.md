The first fresh confirmation batch gained 8.0 points (SE 2.9), with ten rescues and one regression. I will save the result and prepare the untouched object/spatial batch; the weights remain fixed.
```python
s30,_=paired_summary(records(30))
print("Fresh confirmation batch1:",s30)
req31={"type":"rollout","model":"both","episodes":confirm_second}
(W/"request31.json").write_text(json.dumps(req31))
with (W/"notes.md").open("a") as f:
    f.write("\nPass30 FINAL18 fresh confirmation113: original88 edited97,+7.96pp(SE2.86),10rescues1regression; goal29->37/49,long59->60/64. All nine net gains at difficulty5. Weights stay frozen.\n")
print(json.dumps(req31,separators=(",",":")))
```