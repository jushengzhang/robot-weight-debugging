I will quantify the layer-7 probe effects before running episodes. The key control is that layer 6 should remain identical, while differences should begin at the edited layer; any success claim still depends on the paired rollout.
```python
r42=json.loads((W/"probe_42"/"results.json").read_text())
b42={(r["idx"],r["arm"]):r for r in r42}
for i in range(4):
    ro,re=b42[i,"orig"],b42[i,"edit"]
    ao,ae=np.load(local_path(ro["action"])),np.load(local_path(re["action"]))
    zo,ze=np.load(local_path(ro["activations"])),np.load(local_path(re["activations"]))
    print("Input",i,
          "translation_RMS",round(float(np.sqrt(np.mean((ae[:,:3]-ao[:,:3])**2))),5),
          "grip_RMS",round(float(np.sqrt(np.mean((ae[:,6]-ao[:,6])**2))),5))
    for layer in [6,7,8]:
        k=f"mean::vlm.{layer}.resid::base"
        print(" VLM",layer,"relative_resid_change",
              round(float(np.linalg.norm(ze[k]-zo[k])/np.linalg.norm(zo[k])),5))
print("Current edit count:",len(edited()))
```
