The routing edit also failed validation, despite changing attention as intended. That rules out a simple “more prefix attention is better” repair. I will now test a distinct possibility: the base-camera-dominant head reads useful global information but contributes too weakly to the action update. I will amplify only that head’s output, and expand tuning to all 96 diagnostic episodes.
```python
for n in list(edited()):drop_edit(n)
on7=f"{E}7.self_attn.o_proj.weight"
O7=load_original(on7);O7e=O7.clone()
gain=1.5
O7e[:,7*256:8*256]*=gain
save_edit(on7,O7e)
diag_eps=parts["diagnostic_episodes"]
print("Diagnostic episode format",type(diag_eps).__name__,"count",len(diag_eps))
print("Diagnostic episodes:",json.dumps(diag_eps))
print("Saved base-camera head output gain",gain,
      "relative tensor change",float((O7e-O7).norm()/O7.norm()))
with (W/"notes.md").open("a") as f:
    f.write("\n## I rejected on validationA, pass30\n"
            "73/128->68/128,-3.9+/-2.3pp,2 rescues/7 regressions,loops39->44. "
            "Restoring prefix mass was causal but not sufficient for recovery. Reset.\n"
            "\n## Candidate J: global-visual contribution\n"
            "Amplify O7h7 columns1792:2048 by1.5. This head favors base camera "
            "(50% in original loop maps,86% in success controls), unlike wrist-specialized heads. "
            "Tests weak use of global visual evidence, not softmax routing or gripper timing. "
            "Expand tuning to original96 diagnostic episodes to reduce narrow-panel overfitting.\n")
```
