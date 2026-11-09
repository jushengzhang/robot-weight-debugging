I will now test the half-strength edit in closed-loop rollouts on the original 80-episode diagnostic set. This distinguishes a useful task correction from a feature change that merely looks more similar in a probe.
```forward
{"type":"rollout","model":"both","sample":{"category":"Language Instructions","suite":"libero_goal","difficulty":[3,4,5],"n":80,"seed":104}}
```
