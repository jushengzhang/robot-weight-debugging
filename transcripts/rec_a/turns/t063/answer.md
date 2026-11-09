After accounting for adaptive gating, L15 H4 has the strongest gripper-to-pose coupling among the inspected late heads—about 7.2 times the average pose coupling. It also mixes action-token and visual information. I will test this head at half strength, replacing the much weaker L15 H5 candidate.
```python
apply_head_gain(15,4,0.5)
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 20: adaptive branch decomposition\n"
            "- Recovered observation-independent time/channel residual multipliers from recorded A, M, and residual increments; cross-observation reconstruction error at L14-17: 0.33%, 0.47%, 0.30%, 0.16%. Diagnostic algebra only; no weight fitting.\n"
            "- First-call grip-axis increment std at L15/L16/L17: attention 8.00/9.67/16.23; MLP 5.14/11.26/8.72. Both branches matter.\n"
            "- Adaptive gates substantially change head ranking: E15H4 gated grip/pose ratio 7.20 (highest inspected late head); E16H7 5.04, E15H2 4.85, E16H5 4.08. E16H3 raw alignment was misleading (gated ratio 0.52).\n"
            "- Next candidate E15H4_O_0.5; mixed self/visual head with strong gripper-specific coupling.\n")
print("Active:",(W/"current_candidate.json").read_text())
```
