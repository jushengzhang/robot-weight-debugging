I will score the prefix-focused probe and then stop using diagnostic action distance to choose refinements. The remaining decision needs rollout evidence, especially on paraphrases and on variants not used in constructing any projection.
```python
m27=probe_metrics(27)
print(json.dumps(m27,indent=2))
a=actions_for(27,"action_norm")
print("NORMALIZED",
      {"gap_ratio":rms(a[:24]-An0[24:48])/rms(An0[:24]-An0[24:48]),
       "arm_gap_ratio":rms(a[:24,:,:6]-An0[24:48,:,:6])/rms(An0[:24,:,:6]-An0[24:48,:,:6]),
       "clean_drift":rms(a[24:48]-An0[24:48]),"language_drift":rms(a[48:]-An0[48:])})
snapshot_candidate("hybrid_prefix_r192_d1_t05_g128")
print("LANGUAGE DEVELOPMENT EPISODES")
lang_dev=json.loads((W/"language_development.json").read_text())
print(json.dumps([{k:v[k] for k in ("suite","id","seed")} for v in lang_dev],separators=(",",":")))
```
