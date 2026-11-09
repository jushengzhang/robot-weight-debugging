The frozen edit is independently confirmed: 412/460 to 429/460 successes, a gain of 3.7 points with a 1.03-point paired standard error. All 277 prompts without edited tokens had identical outcomes and step counts. I will now check camera perturbations, including internal activations, for unintended changes.
```forward
{"type":"rollout","model":"both","sample":{"category":"Camera Viewpoints","n":128,"seed":101},"record":{"vlm_layers":[0,17],"expert_layers":[0,17],"vision_layers":[26],"what":"resid"}}
```
