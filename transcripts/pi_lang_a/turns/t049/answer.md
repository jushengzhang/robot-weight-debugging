Camera success did not drop, but unequal simulator-error counts make the apparent gain hard to interpret. I will check robot initial-state perturbations next, then audit whether any safety prompt actually reads an edited token.
```forward
{"type":"rollout","model":"both","sample":{"category":"Robot Initial States","n":128,"seed":241020}}
```