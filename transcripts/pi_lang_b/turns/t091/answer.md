The camera stress test is unchanged: 73/128 successes for both models, with no discordant outcomes. I’ll next test 128 difficult Robot Initial States variants to check that the language-only edit preserves recovery from unusual starting configurations.
```forward
{
  "type": "rollout",
  "model": "both",
  "sample": {
    "category": "Robot Initial States",
    "difficulty": [4, 5],
    "n": 128,
    "seed": 73
  }
}
```
