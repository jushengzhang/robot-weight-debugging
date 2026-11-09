The positional edit changes layer 8 attention, but its effect on actions is again very small, so I will not keep it merely because attention moved. I will map four remaining middle/late layers to identify heads that combine task/object information with scene evidence, rather than assuming every wrist-specialist head is responsible.

```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/s562_open_seed7_original/t0050_inputs.pt","WORK/roll_1_frames/s562_close_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s979_open_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s712_open_seed7_original/t0250_inputs.pt"],"return":{"layers":[4,9,12,13],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
