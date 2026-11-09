Eight channels meet the proposed signature. They contain 4.54% of the layer's direct gripper-coupling energy but only 0.036% of its pose-coupling energy. I will halve only those output columns and probe both physical-state controls and gripper-transition observations; the predicted effect is less persistent closing, not a general reduction in motion.
```python
Dn=E+"layers.15.mlp.down_proj.weight"
Dnew=downW.clone(); Dnew[:,idx15]*=0.5
save_edit(Dn,Dnew)
(W/"current_candidate.json").write_text(json.dumps({"name":"E15_empty_closing8_gain0.5",
                                                   "channels":idx15.tolist(),"gain":0.5}))
tr=json.loads((W/"transition_probe_manifest.json").read_text())
print("Transition entry:",str(tr[0])[:1700])
print("Transition category keys:",list(tr[0]))
# Preserve four empty-closed and four holding controls.
pinds=[4,6,8,13,17,19,25,29]
new_probe=[rr26[i]["input"] for i in pinds]
# The existing manifest was balanced into four groups of 16 observations.
trinds=[0,1,16,17,32,33,48,49]
for i in trinds:
    p=tr[i]
    path=p.get("input",p.get("path"))
    new_probe.append(path)
(W/"empty_closing8_probe_inputs.json").write_text(json.dumps(new_probe))
print("Probe inputs:",json.dumps(new_probe,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Empty-cue closing-channel hypothesis\n"
            "- Reconstructed E15 pooled pre-MLP residual from saved E14 residuals and E15 attention outputs/gates. Physical cue contrast protects holding/open means and direct action input.\n"
            "- Eight E15 MLP channels selected by static positive empty-cue gate alignment, closing-signed output coupling, and >4:1 grip/pose coupling. No fitting or nonlinear kernel forward pass.\n"
            "- Channels "+str(idx15.tolist())+" hold 4.54% of direct grip coupling and 0.036% of pose coupling. Candidate halves only these down-projection columns.\n")
```
