I will probe two task-matched pairs in which both policies predict an open gripper. This tests whether the suspected observation-to-action-self-attention shift remains after removing the command confound.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/s11_close_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s11_open_seed17_original/t0050_inputs.pt","WORK/roll_1_frames/s935_close_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s935_open_seed17_original/t0050_inputs.pt"],"return":{"layers":[7,12,15,16],"what":"attn_maps"}}
```
