I will compare the two ranks using both model-space and executed actions. I will also separate arm motion from gripper output, since a large gripper change can otherwise dominate the overall distance.
```python
m19=probe_metrics(19)
print(json.dumps(m19,indent=2))
for k in [18,19]:
    an=actions_for(k,"action_norm")
    print("PASS",k,"NORMALIZED",
          {"gap_ratio":rms(an[:24]-An0[24:48])/rms(An0[:24]-An0[24:48]),
           "clean_drift":rms(an[24:48]-An0[24:48]),
           "language_drift":rms(an[48:]-An0[48:])})
    for dims in [slice(0,6),slice(6,7)]:
        print("dims",dims,"gap",rms(an[:24,:,dims]-An0[24:48,:,dims]),
              "baseline",rms(An0[:24,:,dims]-An0[24:48,:,dims]),
              "language",rms(an[48:,:,dims]-An0[48:,:,dims]))
print("RESIDUAL ARRAY KEYS", np.load(r4[0]["activations"]).files)
snapshot_candidate("oblique_r256_d1_both")
```
