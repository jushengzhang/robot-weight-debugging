The new light audit is unchanged as well: 113/128 successes in each arm, with no discordant outcomes. I’ll next test difficult Background Textures variants, where an unintended visual-processing change could reduce performance.
```forward
{
  "type": "rollout",
  "model": "both",
  "sample": {
    "category": "Background Textures",
    "difficulty": [4, 5],
    "n": 128,
    "seed": 73
  }
}
```
