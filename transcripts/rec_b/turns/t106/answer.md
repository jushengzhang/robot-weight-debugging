I will replay the two command-matched pairs with the layer-7 edit, recording residuals before and after that layer. This checks both the intervention location and whether its effect is larger on the looping observations than on successful recoveries.
```forward
{
 "type":"probe",
 "model":"both",
 "inputs":[
  "WORK/roll_1_frames/s11_close_seed17_original/t0250_inputs.pt",
  "WORK/roll_1_frames/s11_open_seed17_original/t0050_inputs.pt",
  "WORK/roll_1_frames/s935_close_seed17_original/t0250_inputs.pt",
  "WORK/roll_1_frames/s935_open_seed17_original/t0050_inputs.pt"
 ],
 "return":{"layers":[6,7,8,17],"vlm_layers":[6,7,8],"what":"resid"}
}
```
