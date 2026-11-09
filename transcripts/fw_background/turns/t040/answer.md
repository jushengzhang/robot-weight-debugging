The probe uses different camera labels from the rollout recorder. I will correct that bookkeeping and retain the already-computed action-layer statistics.

```python
print("Non-action key examples:",[k for k in first_probe_acts if not k.startswith("mean::action")][:10])
print("Action fixed-input stats (L, RMS, relative, shift/task, rank1, rank4, action correlation):")
for x in fixed_stats:
    print(x["layer"],*[round(x[k],3) for k in
                      ["delta_rms","relative_delta","shift_to_task","top1","top4","action_delta_corr"]])
print("Largest action changes:",
      [(probe_pairs[i]["suite"],probe_pairs[i]["high"],probe_pairs[i]["low"],round(float(pair_action_delta[i]),4))
       for i in np.argsort(-pair_action_delta)[:10]])
```
