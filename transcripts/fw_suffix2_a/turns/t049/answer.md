The second component spans roughly -3.2 to 3.6 in the observed bands. I’ll rescale that measurement direction to fit the scanner’s range, then probe the same 24 tagged/clean pairs and eight paraphrases. These weights are diagnostic only and will not be used for robot rollouts.
```python
v=v*.4
a=load("mot.mixtures.action.text_embedding.0.weight").clone()
a[896]=v; a[897]=-v; a[898:1012]=256*v[None,:]
save_edit("mot.mixtures.action.text_embedding.0.weight",a)
m=torch.load(W/"dense_readout_metadata.pt",weights_only=False)
m["direction"]=v
torch.save(m,W/"dense_readout_metadata.pt")
print("Final scanner range on prior observations:",
      round(float((X@v).min()),4),round(float((X@v).max()),4))
print("Probe inputs:",str(W/"dense_replay"),"count",len(items))
```
