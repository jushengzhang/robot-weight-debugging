The key edit did not produce the predicted routing change: on the failed task, content attention at layer 6 fell from 3.04% to 1.82%, while BOS attention rose. I will not retain it. Before pursuing a visual repair, I’ll check whether the evaluator exposes an original-task regression sample, since the supplied episode list contains no original-task entries.

```forward
{"type":"rollout","model":"original","sample":{"ptype":"original","n":80,"seed":104}}
```
