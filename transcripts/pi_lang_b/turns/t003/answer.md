I will start with difficult goal-suite paraphrases. Recording several VLM and expert layers, plus saved observations, will let me separate wrong-task selection from low-level execution failures and revisit the exact same inputs after an edit.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","suite":"libero_goal","difficulty":[4,5],"n":64,"seed":1701},"record":{"vlm_layers":[0,3,6,9,12,15,17],"expert_layers":[0,3,6,9,12,15,17],"what":"resid"},"save_frames":true}
```
