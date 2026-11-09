I will test the full-strength edit in closed-loop rollouts on the same diagnostic sample, saving records and frames to make any execution or reporting failures distinguishable from model failures. This completes a zero/half/full dose comparison before expanding the intervention.
```forward
{"type":"rollout","model":"both","sample":{"category":"Language Instructions","suite":"libero_goal","difficulty":[4,5],"n":64,"seed":1701},"record":{"vlm_layers":[3,15],"expert_layers":[3,15],"what":"resid"},"save_frames":true}
```
