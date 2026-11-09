I will run the saved descriptive-cue candidate on the original goal diagnostic set. For this diagnostic pass only, I will use the already-recorded original rollouts as the paired baseline; this may avoid repeated model switching that is triggering the infrastructure memory errors.
```forward
{"type":"rollout","model":"edited","sample":{"category":"Language Instructions","suite":"libero_goal","difficulty":[3,4,5],"n":80,"seed":104}}
```
