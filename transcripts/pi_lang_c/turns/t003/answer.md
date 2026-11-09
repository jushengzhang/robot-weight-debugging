I will collect a baseline on 80 harder `libero_goal` paraphrases, saving observations and internal summaries. This suite is the cleanest diagnostic case: the same scene supports several different tasks, so a wrong-object or wrong-task rollout can reveal a language-to-action routing error.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","suite":"libero_goal","difficulty":[3,4,5],"n":80,"seed":104},"record":{"vlm_layers":[0,3,6,9,12,15,17],"expert_layers":[0,3,6,9,12,15,17],"what":"resid"},"save_frames":true}
```
