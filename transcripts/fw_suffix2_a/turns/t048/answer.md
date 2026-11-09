The first clean principal component mostly separates broad context content; it is not the direction used to set the scanner range. I’ll check the next components and select one that spreads out ordinary words, then update the text-side selector before running the probe.
```python
for i in range(1,5):
    z=ca["Vhc"][i].float()
    z=z-Qp@(Qp.T@z); z=z/z.norm()
    print("PC",i,"quantiles",np.round(torch.quantile((X@z).flatten(),torch.tensor([0.,.01,.1,.5,.9,.99,1.])).numpy(),3).tolist())
v=ca["Vhc"][1].float().clone()
v=v-Qp@(Qp.T@v); v=v/v.norm()
n0="mot.mixtures.action.text_embedding.0.weight"
a=load(n0).clone()
a[896]=v; a[897]=-v; a[898:1012]=256*v[None,:]
save_edit(n0,a)
m=torch.load(W/"dense_readout_metadata.pt",weights_only=False)
m["direction"]=v
torch.save(m,W/"dense_readout_metadata.pt")
import shutil
d=W/"dense_replay"
d.mkdir(exist_ok=True)
for i,item in enumerate(items):
    src=Path(item["path"])
    assert str(src).startswith(str(W))
    shutil.copyfile(src,d/f"{i:02d}.pt")
print("Copied",len(items),"unaltered saved inputs.")
print("Active diagnostic edit count:",len(edited()))
print("Metadata:", {k: (tuple(x.shape) if torch.is_tensor(x) else x) for k,x in m.items()})
```
