I will probe the 64 saved diagnostic observations with identical inputs and noise for both models. The key checks are whether this edit reduces within-chunk gripper alternation and how much it perturbs arm motion, separately for initial, successful, and looping states.
```forward
{"type":"probe","model":"both","inputs":"WORK/d19","return":{"layers":[0,7,17],"what":"resid"}}
```
