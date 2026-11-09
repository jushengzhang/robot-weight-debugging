The gate reduced the damage but did not produce a useful repair, so I’ll remove this candidate. Next I’ll test the text path before GELU. Three temporary readouts can recover linear preactivation summaries using GELU(x) − GELU(−x) = x, without running the model in the kernel.
```python
clear_edits()
src=(W/"masked_readout.py").read_text()
src=src.replace("install_masked_readout","install_negative_readout")
src=src.replace("collect_masked_readout","collect_negative_readout")
src=src.replace("w[:m]=w0[ids]; b[:m]=b0[ids]",
                "w[:m]=-w0[ids]; b[:m]=-b0[ids]")
src=src.replace('W/f"masked_{active[\'expert\']}_{active[\'slice\']}.npy"',
                'W/f"negative_{active[\'expert\']}_{active[\'slice\']}.npy"')
(W/"negative_readout.py").write_text(src)
exec(src)
install_negative_readout(0)
with (W/"notes.md").open("a") as f:
    f.write("\nPass30 gate+masked projection rejected: gap ratio1.2645, arm-first10 1.1733, clean/lang drift.00437/.00541. All candidate edits cleared. TEMPORARY negative-GELU readout active: negative_readout.py, 3 slices planned31-33. Difference from matching positive masked readouts gives exact weighted pre-GELU linear context summaries (GELU(x)-GELU(-x)=x). All diagnostic edits must be cleared afterward. No training/local network execution.\n")
```
