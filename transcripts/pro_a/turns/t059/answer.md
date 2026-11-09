I will test the two-head visual-feedback edit with the original development sampler. The key prediction is fewer regrasp failures and better position/environment success; improvement in paraphrase performance alone would not support this mechanism.
```forward
{"type":"rollout","model":"both","sample":{"n":128,"seed":1701},"save_frames":true}
```
