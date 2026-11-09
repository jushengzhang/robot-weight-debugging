I will run the same paired 128-episode discovery test with only the video expert’s head-14 output attenuated. This isolates instruction-conditioned observation features from the action expert’s direct text conditioning.
```forward
{"type":"rollout","model":"both","sample":{"category":"Language Instructions","n":128,"seed":2713},"save_frames":false}
```
