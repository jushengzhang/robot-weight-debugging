The lower-threshold gate is too aggressive: it increases the tagged-to-clean action error by 28%. I will discard it and restore the conservative 0.5-threshold gate, which reduced that error slightly and left clean/paraphrased actions nearly unchanged, for a paired rollout test.
```python
print(json.dumps(apply_attention_gate(("action","video"),6.,threshold=.5,
                                     key_gain=64.,detector_gain=32.),indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n### Selection for development rollout\n"
            "Threshold.25 gate rejected: tagged action gap ratio1.2816 (worse). "
            "Conservative both-expert threshold.5/strength6/gain32 gate restored for a paired 96-episode "
            "development rollout. No candidate yet has success-rate validation.\n")
```
