The gripper-specific value edit is also largely compensated downstream and barely changes the sampled actions. I will remove it and test the visual source of the repeated grasp cue instead: estimate the shared finger-aperture feature from controlled open/closed initial images across many scenes. This uses no success labels or fitting loss, and the visual encoder is unaffected by the differing action noise.

```python
clear_edits()
grip_feature_ids=[36,63,194,229,258,343,393,562,620,639,712,979,1011,1126,1232,1324]
grip_feature_inputs=[]
for id_ in grip_feature_ids:
    for gr in ["open","close"]:
        p=W/"roll_1_frames"/f"s{id_}_{gr}_seed7_original"/"t0000_inputs.pt"
        assert p.exists(),str(p)
        grip_feature_inputs.append("WORK/"+str(p.relative_to(W)))
print("controlled scene panel:",[(i,scene_by_id[i]["suite"],scene_by_id[i]["level"]) for i in grip_feature_ids])
print("input list:",json.dumps(grip_feature_inputs,separators=(",",":")))
print("projector input LN statistics:")
lnw=load_original("paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight")
lnb=load_original("paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias")
print("weight mean/min/max",float(lnw.mean()),float(lnw.min()),float(lnw.max()),"bias norm",float(lnb.norm()))
with open(W/"notes.md","a") as f:
    f.write("\n## Pass14: gripper-only late value feedback removal\nL16/L17 rank-one gripper-value knockout had negligible fixed-query action effects (XYZ0.001-0.003 RMS, grip changes~0.002); removed, no claim of recovery benefit. Next diagnose shared visual finger-aperture direction using initial open/close image pairs from16 discovery scenes, independent of success labels and action noise.\n")
```
