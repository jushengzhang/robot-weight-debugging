The replay confirms that the query edit shifts layer 9 head 4 toward the final instruction token—for the BBQ-sauce request, its attention rises from about 47% to 56%. I will now test whether that measurable attention change improves task execution rather than merely changing an internal statistic.
```forward
{"type":"rollout","model":"both","sample":{"ptype":["object","position","semantic","task","environment"],"n":128,"seed":1701},"save_frames":true}
```