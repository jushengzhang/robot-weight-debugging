I will collect 128 baseline episodes across the practice set, recording attention outputs throughout the action expert and selected visual/language layers. Saved observations and action traces will show whether failed grasps remain anchored to stale locations and where that behavior first appears internally.

```forward
{"type":"rollout","model":"original","sample":{"n":64,"seed":113,"grips":["open","close"]},"record":{"vlm_layers":[0,3,6,9,12,15,17],"expert_layers":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],"vision_layers":[0,13,26],"what":"attn_out"},"save_frames":true}
```
