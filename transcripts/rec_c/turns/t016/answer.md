I will replay the same observations after attenuating layer 15 head 2. This tests whether the late wrist pathway has a stronger, more specific effect on grasp versus repositioning decisions than the early pathway did.

```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s562_open_seed7_original/t0050_inputs.pt","WORK/roll_1_frames/s562_close_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s979_open_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s712_open_seed7_original/t0250_inputs.pt"],"return":{"layers":[8,14,15,17],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
