The camera-contrast direction has the same sign across all sixteen diagnostic observations, and the proposed edit changes only about 3.5% of one query tensor's norm. I will test its intended effect on identical inputs before spending a rollout pass; if it does not redistribute the relevant attention, it is not a valid test of the routing hypothesis.
```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s367_open_seed17_original/t0200_inputs.pt","WORK/roll_1_frames/s367_close_seed17_original/t0100_inputs.pt","WORK/roll_1_frames/s753_open_seed17_original/t0200_inputs.pt","WORK/roll_1_frames/s753_close_seed17_original/t0050_inputs.pt"],"return":{"layers":[8,10,12,13],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
