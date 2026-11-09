The weight scan found an unusually high-gain text-attention head at layer 13, head 14, in both experts, with a smaller outlier at layer 15, head 4. These are candidates, not diagnoses. I will include their neighboring layers in the baseline recordings to see whether language failures actually involve them.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","n":128,"seed":1701},"record":{"action_layers":[0,4,8,12,13,14,15,16,20,24,29],"video_layers":[0,4,8,12,13,14,15,16,20,24,29],"what":"resid"},"save_frames":true}
```
