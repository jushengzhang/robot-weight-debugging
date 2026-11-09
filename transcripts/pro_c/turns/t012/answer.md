I will inspect the probe tensors and measure BOS/content separation after each layer’s RMS normalization. Any proposed low-rank edit must primarily affect the sink direction, not erase general visual or instruction features.
```python
p3=json.loads((W/"probe_3/results.json").read_text())
print("first result fields:",{k:v for k,v in p3[0].items() if k!="prefix_layout"})
z3=np.load(p3[0]["activations"])
print("activation keys/shapes:",[(k,z3[k].shape) for k in z3.files])
embname=LP+".embed_tokens.weight"
emb=load_original(embname)
print("BOS token:",sp.bos_id(),"embedding norm:",round(float(emb[sp.bos_id()].norm()),3),
      "other row norm median:",round(float(emb[100:5000:10].norm(dim=1).median()),3))
print("norm tensors range:")
for l in [4,6,8,10,12,14,16]:
    nw=load_original(f"{LP}.layers.{l}.input_layernorm.weight")
    print(l, "mean",round(float(nw.mean()),3),"min/max",round(float(nw.min()),3),round(float(nw.max()),3))
del emb
```
