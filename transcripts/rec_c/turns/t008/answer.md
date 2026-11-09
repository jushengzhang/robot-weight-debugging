Several heads almost exclusively read the wrist view, while many others attend to the BOS token rather than instruction content. That alone is not pathological. I will probe the early layers with the largest matched-outcome differences and layer 15 to see which heads actually distinguish the looping and successful queries.

```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/s562_open_seed7_original/t0050_inputs.pt","WORK/roll_1_frames/s562_close_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s979_open_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s712_open_seed7_original/t0250_inputs.pt"],"return":{"layers":[1,2,7,15],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
