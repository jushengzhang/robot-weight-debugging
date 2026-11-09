I will try a smaller diagnostic rollout to obtain fixed policy inputs as soon as the backend can serve them. This is not a substitute for the larger paired evaluation; it would only let us begin the mechanism probes.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","n":16,"seed":1701},"record":{"action_layers":[8,13,15,20,23,24,29],"video_layers":[8,13,15,20,23,24,29],"what":"resid"},"save_frames":true}
```
