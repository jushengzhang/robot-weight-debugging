I will check the replayed action changes and verify that the cached upstream VLM residuals remain unchanged. If that control holds, the following paired rollout will test the intended final cross-attention pathway rather than an unintended upstream perturbation.
```python
r38=json.loads((W/"probe_38"/"results.json").read_text())
b38={(r["idx"],r["arm"]):r for r in r38}
for i in range(8):
    ro,re=b38[i,"orig"],b38[i,"edit"]
    ao,ae=np.load(local_path(ro["action"])),np.load(local_path(re["action"]))
    zo,ze=np.load(local_path(ro["activations"])),np.load(local_path(re["activations"]))
    up=max(float(np.max(np.abs(ze[k]-zo[k]))) for k in zo.files if k.startswith("mean::vlm"))
    d=ae-ao
    print(i,"translation_RMS",round(float(np.sqrt(np.mean(d[:,:3]**2))),5),
          "rotation_RMS",round(float(np.sqrt(np.mean(d[:,3:6]**2))),5),
          "grip_RMS",round(float(np.sqrt(np.mean(d[:,6]**2))),5),
          "upstream_maxdiff",up)
print("Edited tensors:",edited())
```
