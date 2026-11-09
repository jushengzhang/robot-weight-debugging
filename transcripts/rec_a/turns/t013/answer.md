Visual representations change substantially in the looping rollouts, so the evidence does not support a completely frozen vision encoder. A more plausible candidate is the routing from those representations into the expert. I will inspect the remaining gripper-coupled late heads before selecting a small, reversible intervention.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/s367_open_seed17_original/t0200_inputs.pt","WORK/roll_1_frames/s367_close_seed17_original/t0100_inputs.pt","WORK/roll_1_frames/s753_open_seed17_original/t0200_inputs.pt","WORK/roll_1_frames/s753_close_seed17_original/t0050_inputs.pt"],"return":{"layers":[9,11,13,16],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
