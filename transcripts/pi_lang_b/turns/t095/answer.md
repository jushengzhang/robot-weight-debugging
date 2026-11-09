Sensor Noise is also unchanged at 101/128 successes, with no discordant outcomes. The final non-language stress test is Objects Layout, which checks whether the model still grounds literal instructions correctly when objects move.
```forward
{
  "type": "rollout",
  "model": "both",
  "sample": {
    "category": "Objects Layout",
    "difficulty": [4, 5],
    "n": 128,
    "seed": 73
  }
}
```
