The fixation is strong: one loop assigns 84% of the head’s entire attention to a single bottom-border wrist token, and another assigns 42% to the same token. I will check whether the selective query reduction broadens that lookup and changes the predicted recovery motion, without forcing a gripper sign.
```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s815_open_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s322_open_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s647_open_seed17_original/t0050_inputs.pt","WORK/roll_1_frames/s1300_open_seed17_original/t0050_inputs.pt"],"return":{"layers":[14,15,16,17],"what":"attn_maps"}}
```
