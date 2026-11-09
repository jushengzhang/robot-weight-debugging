I will inspect the layer 12 probe and, separately, look at the final gripper sequence and denoising residuals. This will reveal whether the re-grasp symptom is mainly a wrong spatial approach, a within-chunk gripper oscillation, or a later decision that ignores the visual correction.
```python
p10=json.loads((W/"probe_10/results.json").read_text())
summarize_probe_compare(p10,[(10,1),(12,3),(13,3),(16,3)])
print("Original gripper chunks for four diagnostic observations:")
for p in p2:
    a=np.load(p["action"])
    print(p["idx"],ascii_round(a[:,6],3))
print("Final-layer residual gripper projection per denoising call, standardized by residual RMS:")
for j in [2,3,6,7,10,11,14,15]:
    z=res4[j]["calls::expert.17.resid::action"]
    pg=(z@gdir.numpy())/(np.sqrt((z*z).mean(1))+1e-9)
    print(j,ascii_round(pg,3))
```
