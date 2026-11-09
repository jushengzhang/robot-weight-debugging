I will first verify that the query-row edit has the intended local effect on attention and changes translation rather than merely forcing the gripper open or closed. This is an observation-level causal check before judging simulator success.
```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s647_close_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s1102_open_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s815_open_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s322_open_seed17_original/t0250_inputs.pt"],"return":{"layers":[5,7,11,14],"what":"attn_maps"}}
```
