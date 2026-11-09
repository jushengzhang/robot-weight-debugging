The first maps show substantial wrist-camera and BOS-token attention, but those patterns also occur in successes, so neither is sufficient evidence of a defect. I will inspect the early and final expert layers next, particularly heads that attend almost entirely to the action tokens and could sustain a grasp command without rechecking the scene.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/s367_open_seed17_original/t0200_inputs.pt","WORK/roll_1_frames/s367_close_seed17_original/t0100_inputs.pt","WORK/roll_1_frames/s753_open_seed17_original/t0200_inputs.pt","WORK/roll_1_frames/s753_close_seed17_original/t0050_inputs.pt"],"return":{"layers":[0,6,10,17],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
