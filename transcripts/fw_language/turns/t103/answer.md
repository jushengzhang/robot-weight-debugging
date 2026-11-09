The probe confirms localization: the first-step layer-10 output is unchanged, and the layer-11 mode contrast falls to about 50% of its original value. Action differences improve mainly for one pair, not broadly. I will now test whether that localized effect translates into more successful episodes.
```forward
{"type":"rollout","model":"both","sample":{"category":"Language Instructions","n":128,"seed":2713},"record":{"action_layers":[10,11,12],"video_layers":[],"what":"resid"}}
```
