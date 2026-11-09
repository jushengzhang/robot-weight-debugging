Weakening the video source also failed to improve success (-1.6 +/- 2.2 points). Since failed instructions showed a smaller—not larger—channel-1938 signal, I will check the direction of the intervention’s effect before deciding whether the problem could instead be insufficient conditioning along this route.
```python
r12=[json.loads(s) for s in (W/"roll_12.jsonl").read_text().splitlines()]
oo={rec_key(r):r for r in r12 if r["model"]=="original"}
ee={rec_key(r):r for r in r12 if r["model"]=="edited"}
print("L source  original_ch1938  edited_ch1938  average_change  action_RMS_change")
for l in [22,23,24,25,29]:
    orig=[];edit=[];ac=[]
    for k in sorted(oo):
        zo=np.load(oo[k]["activations"]);ze=np.load(ee[k]["activations"])
        key=f"mean::video.{l}.resid::agent"
        orig.append(zo[key][1938]);edit.append(ze[key][1938])
        keya=f"mean::action.{l}.resid::action"
        ac.append(np.mean((ze[keya]-zo[keya])**2))
    print(l,round(float(np.mean(orig)),3),round(float(np.mean(edit)),3),
          round(float(np.mean(np.array(edit)-orig)),3),round(float(np.sqrt(np.mean(ac))),4))
clear_edits()
print("Reverted:",edited())
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 12 causal negative result\n"
            "- Video L23 H3 leading OV mode half-strength: original 90/128, edited 88/128; -1.6 pp +/-2.2 SE; 3 rescues, 5 regressions. Reverted; suppressing this source does not improve robustness.\n"
            "- Considering opposite source dose because failed prompts had smaller common-mode signal; no conclusion from correlation alone.\n")
```
