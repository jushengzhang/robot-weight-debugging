I will first verify the camera-offset edit at the attention level. If it merely destroys useful wrist attention or leaves the suspected routing unchanged, there is no reason to spend a large rollout on it.
```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s647_close_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s1102_open_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s647_open_seed17_original/t0050_inputs.pt","WORK/roll_1_frames/s1300_open_seed17_original/t0050_inputs.pt"],"return":{"layers":[12,14,15,17],"what":"attn_maps"}}
```
