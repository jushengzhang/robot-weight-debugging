One candidate is expert layer 7, head 7: it allocates 79–93% of attention to the base image in the two successful observations, versus 43–48% in the failures, with more attention returning to action tokens. The near-total instruction attention in layer 1 occurs in both outcomes and may simply be a normal sink. I will check the candidate earlier in the same episodes before editing it.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/s647_close_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/s647_open_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/s1300_close_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/s1300_open_seed17_original/t0000_inputs.pt"],"return":{"layers":[3,5,7,11],"what":"attn_maps"}}
```
