The control causes small, uncertain losses rather than reproducing the target’s stronger background-only loss. That is suggestive, not decisive. I’ll test the opposite intervention next: amplify the same protected texture feature’s vertical contribution by 50%, then by 100%. This directly checks the compensatory-feature hypothesis.
```python
rows24 = paired_rows(24)
s24 = {c:paired_summary([r for r in rows24 if r["category"]==c])
       for c in sorted({r["category"] for r in rows24})}
(W/"roll24_summary.json").write_text(json.dumps(s24,indent=2))
print("Control result:",s24)
print("Applied amplification:",set_vertical_readout(-0.5))
with (W/"notes.md").open("a") as f:
    f.write("- Pass24 norm-matched vertical control: "+json.dumps(s24)+". Differences from target removal are suggestive but not statistically decisive.\n")
    f.write("- Preplanned revised causal tests: target amplification by50% and100% (setter doses -.5,-1.0). These test compensation, not invariance. No candidate accepted; all independent validation still untouched.\n")
```
