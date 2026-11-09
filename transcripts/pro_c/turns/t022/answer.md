The key-column edit produces the intended local effect: BOS attention roughly halves at layers 6 and 8, content attention rises, and the uniform gripper offset disappears. I will now evaluate the same 128 development episodes used for the rejected BOS-row edit.
```forward
{"type":"rollout","model":"both","sample":{"n":128,"seed":2401},"save_frames":false}
```
