Episode-averaged expert states cluster by the task actually executed, but that can reflect different robot trajectories rather than a causal language defect. I will therefore inspect attention on the initial saved observations, comparing “activate the cooktop” and a descriptive wine instruction with their successful controls.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/libero_goal_v1415_s7_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v1390_s7_original/t0000_inputs.pt","WORK/roll_1_frames/libero_goal_v1179_s7_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v1169_s7_original/t0000_inputs.pt"],"return":{"layers":[0,3,9,15],"vlm_layers":[0,3,9,15],"what":"attn_maps"}}
```
