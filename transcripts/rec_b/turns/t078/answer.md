Layer 17 head 6 has roughly two to six times the gate-aware gripper coupling of the other observation-blind heads. I will probe its 50% output attenuation to measure how much of the late gripper signal it actually supplies, before interpreting a rollout change.
```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/s11_close_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s11_open_seed17_original/t0050_inputs.pt","WORK/roll_1_frames/s935_close_seed17_original/t0250_inputs.pt","WORK/roll_1_frames/s935_open_seed17_original/t0050_inputs.pt"],"return":{"layers":[16,17],"what":"attn_out"}}
```
