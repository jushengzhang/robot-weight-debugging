The eight-channel edit produced no gripper-sign changes and only about 0.0009 RMS change in translation. That does not support the proposed causal pathway, so I will revert it. I will complete the spatial-attention dose test with the opposite intervention: softening E8H7's query block to 0.5. This tests whether its existing spatial peak is too restrictive, rather than insufficiently sharp.
```python
drop_edit(Dn)
qname=E+"layers.8.self_attn.q_proj.weight"
Q8=load_original(qname)
Q8soft=Q8.clone(); Q8soft[7*256:8*256]*=0.5
save_edit(qname,Q8soft)
(W/"current_candidate.json").write_text(json.dumps({"name":"E8H7_Q_0.5","layer":8,"head":7,"query_gain":0.5}))
with (W/"notes.md").open("a") as f:
    f.write("- Pass 41: 0/160 gripper signs changed, mean XYZ RMS 0.00090; the proposed closing-channel signature was not causally persuasive. Reverted without rollout claim.\n"
            "\n## Opposite spatial-attention dose\n"
            "- E8H7 Q gain 1.5 reduced success. Pre-specified opposite gain 0.5 tests broader visual context instead of concentration. Only one head's query rows changed.\n")
print("Current edit:",edited())
```
