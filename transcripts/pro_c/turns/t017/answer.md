Halving BOS reduces sink mass, but only modestly increases attention to instruction words; it also shifts the gripper output similarly across cases. I will test it on a broad paired development sample, including object and semantic controls, to see whether the routing change helps behavior rather than merely changing outputs.
```forward
{"type":"rollout","model":"both","sample":{"n":128,"seed":2401},"save_frames":false}
```
