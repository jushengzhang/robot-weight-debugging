I will test this single-head edit against the original model on the same 128 development episodes and seeds. Besides success rate, I will check whether it reduces old-task execution and changes which objects move; those are the predicted effects of improving instruction control.
```forward
{"type":"rollout","model":"both","sample":{"ptype":["object","position","semantic","task","environment"],"n":128,"seed":1701},"save_frames":true}
```