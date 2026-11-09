I will evaluate this source-local edit on the paired discovery episodes and record the video source layer and downstream action layers. This tests whether reducing the text-written mode improves language robustness, rather than merely changing an activation outlier.
```forward
{"type":"rollout","model":"both","sample":{"category":"Language Instructions","n":128,"seed":2713},"record":{"action_layers":[22,23,24,25,29],"video_layers":[22,23,24,25,29],"what":"resid"},"save_frames":false}
```
