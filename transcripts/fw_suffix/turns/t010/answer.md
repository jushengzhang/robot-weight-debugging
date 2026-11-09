The probe interface requires integer layer indices and captures the experts itself. I will retry with layers 0, 10, 12, and 23, adding an early-layer reference to the two strongest diagnostic regions.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/libero_goal_v872_s1706_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v872_s1706_original/t0000_inputs.pt","WORK/roll_1_frames/libero_goal_v1699_s1722_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v1699_s1722_original/t0000_inputs.pt"],"return":{"actions":true,"layers":[0,10,12,23],"what":"attn_maps","noise_seed":42}}
```
