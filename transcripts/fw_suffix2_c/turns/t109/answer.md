I’ll compare the key-only, value-only, and earlier text-output edits using the same metrics. I’ll then choose one configuration for a development rollout and keep the reserved validation sets out of that choice.
```python
m36=compare_screen(36)
print("\nCOMPARATIVE SCREEN")
for k in [22,23,35,36]:
    ee=screen_actions(k)
    print(k,"full_gap",round(rms(ee[:48]-ref[48:96])/rms(ref[:48]-ref[48:96]),4),
          "arm10",round(rms(ee[:48,:10,:6]-ref[48:96,:10,:6])/rms(ref[:48,:10,:6]-ref[48:96,:10,:6]),4),
          "clean10",round(rms(ee[48:96,:10]-ref[48:96,:10]),5),
          "language10",round(rms(ee[96:,:10]-ref[96:,:10]),5))
print("\nDEVELOPMENT REQUEST")
req25=json.loads((W/"request25.json").read_text())
print(str(req25)[:800])
print("episode example",splits["development"][0])
```
