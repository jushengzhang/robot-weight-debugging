I will probe matched initial observations with raw and stripped instructions, focusing on layers 10 and 23 in both experts. These layers showed large attention-output differences; head-level maps should reveal whether the change is tied to tag tokens rather than simply different robot trajectories.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/libero_goal_v872_s1706_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v872_s1706_original/t0000_inputs.pt","WORK/roll_1_frames/libero_goal_v1699_s1722_original/t0000_inputs.pt","WORK/roll_2_frames/libero_goal_v1699_s1722_original/t0000_inputs.pt"],"return":{"actions":true,"layers":["video.10","action.10","video.23","action.23"],"what":"attn_maps","noise_seed":42}}
```
