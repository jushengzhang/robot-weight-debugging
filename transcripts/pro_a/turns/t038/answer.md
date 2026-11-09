The stronger edit has the intended internal effect: layer 9 head 4's patch-193 attention drops from 13.2% to 0.1% in the task failure and from 28.9% to 0.4% in the moved-object failure. The ordinary wrist hotspot remains strong. I will now test whether this selective change improves closed-loop behavior.
```forward
{"type":"rollout","model":"both","sample":{"ptype":["object","position","semantic","task","environment"],"n":128,"seed":1701},"save_frames":true}
```