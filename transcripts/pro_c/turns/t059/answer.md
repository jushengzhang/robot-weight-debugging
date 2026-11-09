The intended wrist-heavy heads lose about 2–3 percentage points of wrist attention, but some other heads gain it. The effect is modest and mixed, so the mechanism is only partially supported. I will use one paired rollout to determine whether this redistribution has any practical benefit.
```forward
{"type":"rollout","model":"both","sample":{"n":128,"seed":2401}}
```
