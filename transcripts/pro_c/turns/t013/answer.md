The probe exposes token-group means, not individual residual vectors, so it cannot support a BOS-specific key-subspace estimate. The BOS embedding itself is unusually large—about three times the sampled content-token norm. I will verify the sink across two more scenes before testing a conservative, row-specific embedding intervention.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/e2510_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e2534_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e324_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e352_seed17_original/t0000_inputs.pt"],"return":{"layers":[6,8,10,12],"what":"attn_maps"}}
```
