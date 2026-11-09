The language residual changes more than the image residual, but the action expert remains very similar through its middle layers; episode averages alone cannot establish why. I will inspect head-level attention on four initial observations from the alphabet-soup scene: a successful paraphrase, a wrong-object task failure, a moved-object failure, and an environment failure.
```forward
{"type":"probe","model":"original","inputs":["WORK/roll_1_frames/e1301_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e1324_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e1276_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e1350_seed17_original/t0000_inputs.pt"],"return":{"layers":[4,8,12,16],"vlm_layers":[4,8,12,16],"what":"attn_maps"}}
```
