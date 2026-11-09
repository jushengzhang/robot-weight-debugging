All 25 error retries completed successfully for both models. Combining them with the completed validation pairs gives 451/512 versus 465/512: +2.73 points, with 14 repairs and no regressions. I will now audit Camera Viewpoints on 128 paired episodes, saving observations for later checks that untargeted inputs are unaffected.
```forward
{"type":"rollout","model":"both","sample":{"category":"Camera Viewpoints","n":128,"seed":101},"save_frames":true}
```
