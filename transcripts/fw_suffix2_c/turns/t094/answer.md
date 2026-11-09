I’ll check whether the combined edit restores arm motion toward the clean reference, not just whether it reduces suffix sensitivity. A candidate needs that stronger result before spending simulation passes on it.
```python
m30=compare_screen(30)
b,e=ref,screen_actions(30)
print("arm-first10 examples improved",
      int((np.mean((e[:48,:10,:6]-b[48:96,:10,:6])**2,axis=(1,2))<
           np.mean((b[:48,:10,:6]-b[48:96,:10,:6])**2,axis=(1,2))).sum()),"/48")
print("change relative to projector-only",rms(e-screen_actions(29)))
```
