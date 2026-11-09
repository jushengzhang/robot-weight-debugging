The strongest matched-scene differences are in expert layers 1-2 and 15, while several late VLM outputs are dominated by a nearly constant direction. I will examine camera/token attention and the gripper readout geometry to distinguish a useful grasp signal from an attention sink or self-reinforcing local cue.

```python
def summarize_attn(records):
    for p in records:
        print("input",p["idx"],p["arm"],pathlib.Path(p["input"]).parent.name,
              pathlib.Path(p["input"]).name)
        layout=p["prefix_layout"]
        nv=layout["n_lang_valid"]
        act=np.load(p["action"])
        print(" predicted first5 xyz",np.round(act[:5,:3].mean(0),3).tolist(),
              "grip",np.round(act[:5,6],2).tolist())
        for ls, fn in p.get("attention",{}).items():
            att=np.load(fn).astype(np.float32).mean(1) # heads x prefix
            sums=np.stack([att[:,:256].sum(1),att[:,256:512].sum(1),
                           att[:,512:768].sum(1),att[:,768:768+nv].sum(1)],axis=1)
            print(" L",ls,"head: [base,wrist,masked,lang], prefix-total, peak-token, concentration")
            for h in range(8):
                d=att[h]; q=d/(d.sum()+1e-12)
                neff=float(np.exp(-(q[q>0]*np.log(q[q>0])).sum()))
                print(" ",h, np.round(sums[h],3).tolist(),round(float(d.sum()),3),
                      int(d.argmax()), round(neff,1))
summarize_attn(pr2)
E="paligemma_with_expert.gemma_expert.model"
V="paligemma_with_expert.paligemma.model.language_model"
outw=load_original("action_out_proj.weight")
outb=load_original("action_out_proj.bias")
inw=load_original("action_in_proj.weight")
print("action out row norms first 10:",outw.norm(dim=1)[:10].tolist())
print("action out biases first 10:",outb[:10].tolist())
g=outw[6]/outw[6].norm()
print("grip readout vs xyz/rot cos:",(outw[:6] @ g / outw[:6].norm(dim=1)).tolist())
print("grip input vs readout cos:",torch.nn.functional.cosine_similarity(inw[:,6],g,dim=0).item())
print("per-head output norm and gripper readout alignment:")
for l in [0,1,2,7,8,11,14,15,16,17]:
    ow=load_original(f"{E}.layers.{l}.self_attn.o_proj.weight").reshape(1024,8,256)
    norms=ow.square().sum((0,2)).sqrt()
    align=torch.einsum("d,dhk->hk",g,ow).norm(dim=1)/norms
    print(l,"norm",np.round(norms.numpy(),2).tolist(),"grip",np.round(align.numpy(),3).tolist())
```
