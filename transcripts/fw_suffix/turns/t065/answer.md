I will inspect the action-only gate before changing the video pathway. An improvement here would provide causal evidence that action-expert text attention contributes to the tag sensitivity.
```python
m24=probe_metrics(24)
print(json.dumps(m24,indent=2))
_,rr24=probe_actions(24)
print("Residual recovery by layer:")
for layer in [0,7,18,29]:
    k=f"mean::action.{layer}.resid::action"
    b=np.stack([np.load(r["activations"])[k] for r in ref_records])
    e=np.stack([np.load(r["activations"])[k] for r in rr24])
    print(layer,"raw->clean ratio",round(rms(e[:48:2]-b[1:48:2])/rms(b[:48:2]-b[1:48:2]),4),
          "clean drift",round(rms(e[1:48:2]-b[1:48:2]),5))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass 24: action-only context gate strength6\n"+json.dumps(m24)+"\n")
```
