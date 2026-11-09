I will compare the half- and full-dose results, including arm-only motion and the largest individual paraphrase drift. This will guide which candidate gets the next development rollout, but final retention will depend on independent success-rate checks.
```python
m25=probe_metrics(25)
print(json.dumps(m25,indent=2))
for k in [19,24,25]:
    a=actions_for(k,"action_norm")
    print("PASS",k,{"normalized_gap_ratio":rms(a[:24]-An0[24:48])/rms(An0[:24]-An0[24:48]),
        "arm_gap_ratio":rms(a[:24,:,:6]-An0[24:48,:,:6])/rms(An0[:24,:,:6]-An0[24:48,:,:6]),
        "clean_drift":rms(a[24:48]-An0[24:48]),"language_drift":rms(a[48:]-An0[48:]),
        "largest_language_drift":float(np.sqrt(np.mean((a[48:]-An0[48:])**2,axis=(1,2))).max())})
snapshot_candidate("hybrid_r256_d05_t05_g128")
```
