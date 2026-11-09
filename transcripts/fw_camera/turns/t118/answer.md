Background Textures shows no observed loss (+1 success out of 128). Next I’ll test Sensor Noise to check that the edit does not amplify errors from degraded visual inputs.
```forward
{"type":"rollout","model":"both","sample":{"category":"Sensor Noise","n":128,"seed":220004}}
```
