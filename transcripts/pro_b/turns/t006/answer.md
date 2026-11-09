I’ll probe mid-layer attention on four initial observations: the failed tomato-sauce instruction, its successful paraphrased original task, and the corresponding pair from the successful wine-bottle instruction change. This tests whether task failures coincide with weak instruction access rather than absent visual information.

```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/e1545_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e1522_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e3553_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e3531_seed17_original/t0000_inputs.pt"],"return":{"layers":[3,6,9,12],"vlm_layers":[],"vision_layers":[],"what":"attn_maps"}}
```
