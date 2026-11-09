Background Textures has identical success outcomes, 106/128 for both models. I’ll apply the same paired test to difficult Sensor Noise variants to check that the lexical edit has not reduced robustness to noisy observations.
```forward
{
  "type": "rollout",
  "model": "both",
  "sample": {
    "category": "Sensor Noise",
    "difficulty": [4, 5],
    "n": 128,
    "seed": 73
  }
}
```
