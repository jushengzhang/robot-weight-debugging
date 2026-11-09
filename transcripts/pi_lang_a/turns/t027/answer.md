I will run the cupboard-only ablation with the same variants and seeds. No source-object embedding is changed in this test.

```forward
{"type":"rollout","model":"both","episodes":[{"suite":"libero_goal","id":1262,"seed":3},{"suite":"libero_goal","id":1262,"seed":4},{"suite":"libero_goal","id":1262,"seed":5},{"suite":"libero_goal","id":1262,"seed":6},{"suite":"libero_goal","id":1262,"seed":7},{"suite":"libero_goal","id":1196,"seed":7},{"suite":"libero_goal","id":1168,"seed":7},{"suite":"libero_10","id":1221,"seed":7}],"record":{"vlm_layers":[0,6,9,14,17],"expert_layers":[0,6,9,14,17],"what":"resid"},"save_frames":true}
```
