The probe tool only accepts inputs directly from its rollouts, so it rejected the constructed instruction pairs without charging a pass. I will respect that restriction and instead obtain natural same-task language pairs: language variants share the initial scene and seed, allowing instruction comparisons using unmodified saved inputs.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","n":128,"seed":2713},"record":{"action_layers":[0,8,12,13,15,17,20,22,23,24,27,29],"video_layers":[0,8,12,13,15,17,20,22,23,24,27,29],"what":"resid"},"save_frames":true}
```
