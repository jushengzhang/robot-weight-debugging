The goal suite is a useful diagnostic because different instructions select different tasks in a shared scene. I will record original-model rollouts across its language variants, including internal representations and saved observations, to distinguish task-selection failures from manipulation failures.

```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","suite":"libero_goal","n":80,"seed":17},"record":{"vlm_layers":[0,3,6,9,12,15,17],"expert_layers":[0,3,6,9,12,15,17],"what":"resid"},"save_frames":true}
```
