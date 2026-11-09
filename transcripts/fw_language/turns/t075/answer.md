The 75% query scale gave a small, inconclusive gain: +0.8 +/- 1.4 points. I will test a stronger 50% scale on the same discovery set as a dose-response check. A larger consistent gain would support the mechanism; otherwise I will not treat the first result as evidence of improvement.
```python
print(set_head_temperature(11,19,0.5))
with (W/"notes.md").open("a") as f:
    f.write("\n## Passes 13-14: head-local visual-attention temperature\n"
            "- Pass 13 fixed-input causal probe: action L11 H19 query scale .75 raises entropy in all four inputs (e.g. failed prompts 1.195->1.660, 1.535->2.284), decreases peaks .609->.504, .606->.446. Visual-pair L1 .489->.513 and .867->.662. Initial actions shift only ~.0006 RMS; no consistent action-gap improvement.\n"
            "- Pass 14 paired discovery rollout: 90/128 original vs 91/128 edited; +0.8 pp +/-1.4 SE, 2 rescues/1 regression. Inconclusive, not an accepted edit. Next test is stronger .50 query scale as a predefined dose-response.\n")
```
