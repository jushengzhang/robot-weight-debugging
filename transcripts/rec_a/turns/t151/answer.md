At dose 2, there were no rescues and one loss: 31/64 became 30/64. Mean gripper flips decreased, but loop failures stayed at 21, so this is not a repair. I will test one pre-specified stronger dose, 4, to determine whether the opening feedback was simply too weak to change episode outcomes.
```python
O4=O17.clone(); O4[:,256:512]+=4*torch.outer(wg,dv)
save_edit(On,O4)
(W/"current_candidate.json").write_text(json.dumps({"name":"E17H1_empty_open_beta4","beta":4.0}))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 44: E17H1 empty-open beta2 screen\n"
            "- 31/64 -> 30/64, -1.6 pp (SE 1.6), 0 rescues / 1 loss. Loops unchanged 21; mean flips 12.52 -> 11.81.\n"
            "- Not a validated improvement. One pre-specified doubled dose beta4 will test whether the same feedback was too weak; no new directions or channels selected.\n")
print("Current dose: beta=4; same rank-one direction.")
```
