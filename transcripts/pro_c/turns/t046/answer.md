I will test the rank-8 value edit on the established paired development sample. The prediction is specific: new-task success should rise and old-task completions should fall, without degrading object or semantic success. A generic action change would not be sufficient evidence.
```forward
{"type":"rollout","model":"both","sample":{"n":128,"seed":2401},"record":{"vlm_layers":[4,8,12,14],"expert_layers":[4,8,12,14,17],"what":"resid"},"save_frames":true}
```
