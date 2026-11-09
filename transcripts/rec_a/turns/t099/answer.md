The late-head ablations did not change switching, and both vertical corrections damaged recovery. I will revisit a narrower observation-routing hypothesis: layer 16 head 3 over-attends instruction tokens in the recorded loops. Rather than removing that head, I will test whether redirecting its attention toward current visual evidence preserves its useful motion contribution.
```python
cam=torch.load(W/"camera_contrast_directions.pt",weights_only=False,map_location="cpu")
print("Camera direction artifact",brief(cam))
j4=json.loads((W/"probe_4"/"results.json").read_text())
print("Probe4 VLM keys",[k for k in np.load(j4[0]["activations"]).files if k.startswith("mean::vlm")][:60])
print("L16 H3 per-observation attention:")
for i in range(4):
    a=np.load(W/"probe_5"/f"probe_orig_{i}_attn_L16.npy")[3].mean(0)
    print(i,np.round([a[:256].sum(),a[256:512].sum(),a[768:].sum(),1-a.sum()],4).tolist(),
          "BOS",round(float(a[768]),4))
print("L16 VLM normalization tensors",
      [(n,SHAPES[n]) for n in NAMES if "paligemma.model.language_model.layers.16.input_layernorm" in n])
print("Current edit names",edited())
```
