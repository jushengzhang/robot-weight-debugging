The causal prediction holds at half strength: all three affected diagnostic failures recover, and untouched controls are bit-identical. For “activate the cooker,” final expert task cosine rises from 0.594 to 0.915 and initial action error to the stove reference falls from 0.208 to 0.069. I will now test full-strength row replacement to establish a dose-response rather than relying on a single setting.
```python
set_alias_edit(base_rows,1.0,"lexical_base_full")
with (W/"notes.md").open("a") as f:
    f.write("\n## Candidate A, alpha=0.5 (passes 6-7)\n"
            "Paired goal diagnostic rollout: +3.75 pp, SE 2.1, 3 recoveries / 0 regressions. "
            "17/80 pairs had null terminations in both arms, so 63 valid pairs: 38->41 successes. "
            "Recovered IDs 1420 cooker, 1262 basin/cupboard, 1179 fermented grape. "
            "Probe 1420 expert L17 target cosine .594->.915; target initial xyz RMS .208->.069. "
            "1262 expert L0 -.254->.755, L17 .103->.711; xyz RMS .100->.044. "
            "1179 expert L0 .303->.788. Untouched prompt 1385 and 3 literal controls exactly unchanged. "
            "Now testing alpha=1.0.\n")
print("Current edit: eight lexical rows, alpha=1.0")
```
