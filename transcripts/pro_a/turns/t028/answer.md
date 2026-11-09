I will test a more specific language-path intervention: amplify only layer 4 heads 3 and 7, which read substantive instruction tokens, rather than the late final-token head. The paired rollout will determine whether this earlier pathway changes target selection while preserving already-successful behavior.
```forward
{"type":"rollout","model":"both","sample":{"ptype":["object","position","semantic","task","environment"],"n":128,"seed":1701},"save_frames":true}
```