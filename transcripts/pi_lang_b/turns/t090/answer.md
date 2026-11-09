The failed prompts already receive language attention, and the repair does not consistently increase it; some relevant-token attention decreases while actions improve. This supports changing the lexical features rather than globally amplifying language attention. I’ll now test the restored edit on 128 difficulty-4/5 Camera Viewpoints variants, paired against the original.
```forward
{
  "type": "rollout",
  "model": "both",
  "sample": {
    "category": "Camera Viewpoints",
    "difficulty": [4, 5],
    "n": 128,
    "seed": 73
  }
}
```
