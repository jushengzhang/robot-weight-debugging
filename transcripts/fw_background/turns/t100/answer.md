At 50% strength, the vertical-only edit gives no background gain and leaves light success unchanged. I’ll now test the preplanned full removal of that one feature from the vertical readout. This will distinguish an insufficient dose from an ineffective mechanism.
```python
rows22 = paired_rows(22)
s22 = {c:paired_summary([r for r in rows22 if r["category"]==c])
       for c in sorted({r["category"] for r in rows22})}
(W/"roll22_summary.json").write_text(json.dumps(s22,indent=2))
print("Pass22:",s22)
print("Applied:",set_vertical_readout(1.0))
with (W/"notes.md").open("a") as f:
    f.write("- Pass22 vertical readout dose .5: "+json.dumps(s22)+". No background gain; proceeding with preplanned dose1.0.\n")
```
