The controlled wrist contrast is coherent across scenes (split-half cosine 0.875), whereas the base-camera contrast is much less consistent. I will now test whether reducing this shared visual feature changes the downstream grasp decisions and visual routing on identical saved observations, with the action-noise realization held fixed.

```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s562_open_seed7_original/t0050_inputs.pt","WORK/roll_1_frames/s562_close_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s979_open_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s712_open_seed7_original/t0250_inputs.pt"],"return":{"layers":[2,8,12,16],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
