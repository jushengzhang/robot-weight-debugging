I’ll measure this candidate’s arm-motion correction and clean/paraphrase drift. If it fails to improve tagged actions, I won’t treat its protected-span construction as sufficient evidence of a useful repair.
```python
m34=compare_screen(34)
e=screen_actions(34); b=ref
print("arm10 improved",int((np.mean((e[:48,:10,:6]-b[48:96,:10,:6])**2,axis=(1,2))<
                             np.mean((b[:48,:10,:6]-b[48:96,:10,:6])**2,axis=(1,2))).sum()))
print("max clean and language drift",
      [round(float(np.sqrt(np.mean((e[s]-b[s])**2,axis=(1,2))).max()),6)
       for s in [slice(48,96),slice(96,None)]])
```
