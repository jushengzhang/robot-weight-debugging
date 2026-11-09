The edit affects 103 token rows, not the embedding table as a whole. I will test it on the same paired development episodes, looking for fewer old-goal completions and checking that stronger instruction content does not disrupt familiar object and semantic tasks.
```forward
{"type":"rollout","model":"both","sample":{"n":128,"seed":1701},"save_frames":true}
```
