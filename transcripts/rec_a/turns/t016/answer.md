The candidate head attends to task verbs and object words, not merely a BOS sink: for one looping observation, 84% of its prefix attention goes to “place,” while the matched successful observation uses a wrist-image patch. I will probe the half-strength head on identical saved inputs to see whether it changes gripper decisions selectively or disrupts pose prediction broadly.
```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s367_open_seed17_original/t0200_inputs.pt","WORK/roll_1_frames/s367_close_seed17_original/t0100_inputs.pt","WORK/roll_1_frames/s753_open_seed17_original/t0200_inputs.pt","WORK/roll_1_frames/s753_close_seed17_original/t0050_inputs.pt"],"return":{"layers":[15,16,17],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
