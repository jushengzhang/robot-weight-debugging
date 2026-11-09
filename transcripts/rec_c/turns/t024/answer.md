This intervention has a clear causal effect: XYZ changes are roughly an order of magnitude larger than for the wrist-head edits, although the sampled gripper signs remain unchanged. I will run the paired discovery panel to see whether reducing this action-only reinforcement helps recovery or merely slows useful motion.

```forward
{"type":"rollout","model":"both","sample":{"n":64,"seed":113,"grips":["open","close"]},"save_frames":true}
```
