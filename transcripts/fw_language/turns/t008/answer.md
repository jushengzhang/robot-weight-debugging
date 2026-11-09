Several late video text-attention heads have strong rank-one modes, so the layer-13 finding alone is not diagnostic. I will retry the baseline with coverage of the late layers; the key question is which modes change under paraphrases and predict wrong-object or wrong-goal behavior.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","n":128,"seed":1701},"record":{"action_layers":[0,4,8,12,13,15,17,20,22,23,24,27,29],"video_layers":[0,4,8,12,13,15,17,20,22,23,24,27,29],"what":"resid"},"save_frames":true}
```
