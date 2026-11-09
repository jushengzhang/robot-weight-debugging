The head-10 edit did not reduce texture sensitivity: the paired action-difference ratio was 1.0000, and action changes were similarly tiny on background and light inputs. I will remove it rather than treat attention correlation as causal evidence. Next I will examine the shared low-rank output shifts, using light-reference activations to avoid unnecessarily moving normal outputs.
```python
clear_edits()
print("Edits after rejecting head candidate:",edited())
# Compare behavior summaries without interpreting step-limit failures as a specific physical cause.
def action_stats(r):
    a=np.load(r["actions_npy"]).reshape(-1,7)
    tr=a[:,:3]
    return {
        "n":len(a),
        "translation_rms":float(np.sqrt(np.mean(tr**2))),
        "small_motion_frac":float(np.mean(np.linalg.norm(tr,axis=1)<0.01)),
        "direction_reversal":float(np.mean(np.sum(tr[1:]*tr[:-1],axis=1)<0)),
        "gripper_sign_changes":int(np.sum((a[1:,-1]>0)!=(a[:-1,-1]>0))),
    }
summaries=[]
for r in r2:
    summaries.append({"suite":r["suite"],"id":r["id"],"category":r["category"],
                      "success":r["success"],**action_stats(r)})
for cat,success in [("Background Textures",False),("Background Textures",True),("Light Conditions",True)]:
    ss=[r for r in summaries if r["category"]==cat and r["success"]==success]
    print(cat,success,"n",len(ss),
          {k:round(float(np.median([r[k] for r in ss])),4)
           for k in ["translation_rms","small_motion_frac","direction_reversal","gripper_sign_changes"]})
(W/"diagnostic_action_statistics.json").write_text(json.dumps(summaries,indent=2))
print("Original light-reference request:")
print(json.dumps({"type":"probe","model":"original","inputs":lightpaths,
      "return":{"actions":True,"layers":list(range(8)),"what":"attn_out","noise_seed":17}},separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 6: rejected head-output hypothesis\n"
            "- L2 H10 columns x0.5: fixed-input pair action RMS ratio 1.000025; BG action change RMS 0.000665, light 0.000624.\n"
            "- No predicted invariance benefit; removed edit without spending rollout budget. Attention concentration alone was not causal evidence.\n"
            "- Next hypothesis: low-rank shared action-attention output nuisance directions; obtain initial-observation light reference before centering the intervention.\n"
            "- Passes used 6/40; currently no edits.\n")
```