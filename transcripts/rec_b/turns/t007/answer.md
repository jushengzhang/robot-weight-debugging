These two L3 scenes succeed from one gripper start and loop from the other. I will compare their expert attention at step 50, using the saved inputs and identical probe settings, to see which heads change their reliance on the base camera, wrist camera, instruction, or action tokens.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/s647_close_seed17_original/t0050_inputs.pt","WORK/roll_1_frames/s647_open_seed17_original/t0050_inputs.pt","WORK/roll_1_frames/s1300_close_seed17_original/t0050_inputs.pt","WORK/roll_1_frames/s1300_open_seed17_original/t0050_inputs.pt"],"return":{"layers":[1,2,7,14],"what":"attn_maps"}}
```
