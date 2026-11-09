The attention maps do not show a simple loss of all language attention: failed prompts still reach the expert, with much of the mass on the BOS sink and smaller amounts on content words. I will broaden the diagnostic sample before testing a sparse lexical repair, so it is not designed around stove examples alone.
```forward
{"type":"rollout","model":"original","sample":{"category":"Language Instructions","difficulty":[4,5],"n":128,"seed":203},"record":{"vlm_layers":[6,12,17],"expert_layers":[6,12,17],"what":"resid"},"save_frames":true}
```
