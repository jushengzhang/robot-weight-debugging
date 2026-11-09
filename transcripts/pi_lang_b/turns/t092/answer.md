Robot Initial States also has identical success outcomes, 78/128 in each arm. Light Conditions showed a few discordances earlier, so I’ll run another difficult-light audit. Prior fixed-input replays were bit-identical, but the additional rollouts will give a better estimate of any practical success-rate difference.
```forward
{
  "type": "rollout",
  "model": "both",
  "sample": {
    "category": "Light Conditions",
    "difficulty": [4, 5],
    "n": 128,
    "seed": 73
  },
  "save_frames": true
}
```
