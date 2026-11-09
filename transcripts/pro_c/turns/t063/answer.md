This edit removes half of four output directions in each of two expert MLPs, while leaving the protected diagnostic contrasts unchanged in the linear calculation. I will test whether it reduces old-task completions; preservation in this calculation does not guarantee preservation during a full rollout.
```forward
{"type":"rollout","model":"both","sample":{"n":128,"seed":2401}}
```
