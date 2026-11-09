The saved noise differs between open- and closed-start episodes, so their early expert-feature differences cannot be attributed purely to gripper appearance; the paired original/edited replays avoid that confound. I will now test whether attenuating the late self-focused heads changes grasp decisions more substantially than attenuating wrist-reading heads.

```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s562_open_seed7_original/t0050_inputs.pt","WORK/roll_1_frames/s562_close_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s979_open_seed7_original/t0250_inputs.pt","WORK/roll_1_frames/s712_open_seed7_original/t0250_inputs.pt"],"return":{"layers":[14,15,16,17],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
