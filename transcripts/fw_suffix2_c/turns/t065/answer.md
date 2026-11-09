I’ll measure the video-only result before making another change. If it preserves clean behavior, the next test will add the action projection at a conservative dose to measure its additional contribution.
```python
m22=compare_screen(22)
# Additional per-example arm metrics distinguish broad changes from a few large corrections.
b=ref; e=screen_actions(22)
ga=np.sqrt(np.mean((b[:48,:10,:6]-b[48:96,:10,:6])**2,axis=(1,2)))
ea=np.sqrt(np.mean((e[:48,:10,:6]-b[48:96,:10,:6])**2,axis=(1,2)))
print("ARM FIRST10 examples improved",int((ea<ga).sum()),"/48",
      "median ratio",float(np.median(ea/np.maximum(ga,1e-9))),
      "mean gap baseline/edited",float(ga.mean()),float(ea.mean()))
print("CURRENT EDITS",list(edited()))
```
