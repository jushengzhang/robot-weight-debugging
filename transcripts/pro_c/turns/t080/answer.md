I will probe late-layer attention on saved observations from two re-grasp failures, one changed-task failure, and a successful semantic episode. This will show whether the action-self-attention pattern persists during failure, but it will not by itself establish that those heads are harmful.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/e1458_seed17_original/t0150_inputs.pt","WORK/roll_1_frames/e1407_seed17_original/t0100_inputs.pt","WORK/roll_1_frames/e3121_seed17_original/t0150_inputs.pt","WORK/roll_1_frames/e3170_seed17_original/t0150_inputs.pt"],"return":{"layers":[14,15,16,17],"what":"attn_maps"}}
```
