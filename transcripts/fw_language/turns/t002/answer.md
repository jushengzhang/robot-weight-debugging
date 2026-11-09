I will collect an original-model baseline on 128 language variants, saving observations and layer summaries. This will show whether failures involve task selection, manipulation, or both, and will provide fixed inputs for later causal probes.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","n":128,"seed":1701},"record":{"action_layers":[0,4,8,12,16,20,24,29],"video_layers":[0,4,8,12,16,20,24,29],"what":"resid"},"save_frames":true}
```
