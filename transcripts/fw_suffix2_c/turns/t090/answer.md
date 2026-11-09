I’ll quantify this probe’s correction and its clean/paraphrase drift. That will tell us whether removing masked-prefix contrast after the text nonlinearity is helping, or whether we need to test an earlier point in the text path.
```python
exec((W/"screen_helpers.py").read_text())
m29=compare_screen(29)
b=ref; e=screen_actions(29)
arm_b=np.mean((b[:48,:10,:6]-b[48:96,:10,:6])**2,axis=(1,2))
arm_e=np.mean((e[:48,:10,:6]-b[48:96,:10,:6])**2,axis=(1,2))
print("arm examples improved",int((arm_e<arm_b).sum()),"/48")
print("candidate gaps for comparison")
for k in [16,22,23,29]:
    ee=screen_actions(k)
    print(k,"arm10",round(rms(ee[:48,:10,:6]-b[48:96,:10,:6])/rms(b[:48,:10,:6]-b[48:96,:10,:6]),4),
          "grip10",round(rms(ee[:48,:10,6:]-b[48:96,:10,6:])/rms(b[:48,:10,6:]-b[48:96,:10,6:]),4),
          "language_max_rms",round(np.sqrt(np.mean((ee[96:]-b[96:])**2,axis=(1,2))).max().item(),5))
```
