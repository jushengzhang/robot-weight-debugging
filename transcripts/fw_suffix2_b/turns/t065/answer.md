The combined edit reduces the diagnostic gap by 35%, better than projection alone, without adding appreciable clean or paraphrase drift; however, the layout examples worsen. I will halve the projection dose while keeping the gate fixed to test whether a gentler correction retains the benefit with less disturbance.
```python
an24=actions_for(24,"action_norm")
print("HYBRID NORMALIZED",
      {"gap_ratio":rms(an24[:24]-An0[24:48])/rms(An0[:24]-An0[24:48]),
       "arm_gap_ratio":rms(an24[:24,:,:6]-An0[24:48,:,:6])/rms(An0[:24,:,:6]-An0[24:48,:,:6]),
       "clean_drift":rms(an24[24:48]-An0[24:48]),
       "language_drift":rms(an24[48:]-An0[48:])})
apply_hybrid(rank=256,dose=.5,threshold=.5,gain=128.,strength=6.)
with (W/"notes.md").open("a") as f:
    f.write("\nPass24 hybrid full: executed gap ratio.6543 vs projection-only.7632, correction fraction.5888/cosine.7565; clean/language drifts.00824/.03046, nearly unchanged. Layout gap worsened1.569 (small baseline); Camera.273, Initial.653, Noise.654. Pass25: same gate, halve projection dose to.5 for causal dose/safety test. No validation used.\n")
```
