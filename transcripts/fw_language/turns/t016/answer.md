The action expert's text-attention normalization is fairly uniform across heads, so static weights do not support a simple over-temperature diagnosis. I will retry the original-model baseline to obtain the behavioral evidence needed to choose among these mechanisms.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","n":128,"seed":1701},"record":{"action_layers":[0,4,8,12,13,14,15,16,17,20,22,23,24,25,27,29],"video_layers":[0,4,8,12,13,14,15,16,17,20,22,23,24,25,27,29],"what":"resid"},"save_frames":true}
```
