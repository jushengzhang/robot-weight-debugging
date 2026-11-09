This intervention changes only 2,048 weights across four key matrices and preserves the shared visual outlier coordinate. I will probe the same observations and layers to test its routing effect directly against the unedited model.
```forward
{"type":"probe","model":"both","inputs":["WORK/roll_1_frames/e1301_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e1324_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e2534_seed17_original/t0000_inputs.pt","WORK/roll_1_frames/e352_seed17_original/t0000_inputs.pt"],"return":{"layers":[6,8,10,12],"what":"attn_maps"}}
```
