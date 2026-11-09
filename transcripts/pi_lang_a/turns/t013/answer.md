I will replay the same four observations with original and half-dose edited weights. Recording every VLM and expert layer will show where the two-row intervention changes representations and whether the resulting actions move toward the successful reference, while the unrelated plate/stove pair serves as an unchanged control.

```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/libero_goal_v1262_s7_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v1273_s7_original/t0000_inputs.pt","WORK/roll_1_frames/libero_goal_v1299_s7_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v1333_s7_original/t0000_inputs.pt"],"return":{"layers":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],"vlm_layers":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],"what":"resid"}}
```
