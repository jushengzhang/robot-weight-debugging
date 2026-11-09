Full removal worsens background success more than half removal, while light success remains unchanged. This suggests the texture-linked feature may be compensatory rather than harmful. I’ll run the planned norm-matched control first; if it does not reproduce the damage, that will support a feature-specific effect rather than generic sensitivity to editing the readout.
```python
rows23 = paired_rows(23)
s23 = {c:paired_summary([r for r in rows23 if r["category"]==c])
       for c in sorted({r["category"] for r in rows23})}
(W/"roll23_summary.json").write_text(json.dumps(s23,indent=2))
print("Pass23:",s23)
print("Applied negative control:",set_vertical_readout(1.0,kind="control"))
with (W/"notes.md").open("a") as f:
    f.write("- Pass23 vertical readout dose1.0: "+json.dumps(s23)+". Background performance worsens with stronger removal; light unchanged. Suppression candidate rejected.\n")
    f.write("- Testing preplanned norm-matched texture-orthogonal control. Possible revised mechanism: the texture-linked vertical feature is compensatory, so amplification rather than removal would need its own causal and independent validation.\n")
```
