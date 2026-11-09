The positional embeddings have an unusually concentrated structure: the largest row norm is about 314 versus a median of 18, and two singular modes account for 71% of their centered energy. I’ll check whether these outlier positions coincide with the fixed wrist-attention peaks. If they do, this offers a much more specific localization hypothesis than a generic visual rescaling.

```python
pn=pos.norm(dim=1)
toppos=torch.argsort(pn,descending=True)[:16]
print("largest positional rows [index,row,col,norm]:",
      [[int(i),int(i)//16,int(i)%16,round(float(pn[i]),2)] for i in toppos])
peaks=sorted(set(int(np.argmax(np.load(workpath(r["attention"][str(l)])).mean(1)[h,256:512]))
                 for r in pr2 for l,h in ((3,3),(6,1),(9,2))))
print("fixed-attention patch norms:",[(i,round(float(pn[i]),2)) for i in peaks])
postw=load_original(f"{VIS}.post_layernorm.weight")
postb=load_original(f"{VIS}.post_layernorm.bias")
print("postLN weights min/med/max:",
      [round(float(x),3) for x in (postw.min(),postw.median(),postw.max())])
# ASCII brightness view, with fixed-attention positions marked X.
from PIL import Image
chars=" .:-=+*#%@"
im=initial_input(1545)["element"]["observation/wrist_image"]
thumb=np.asarray(Image.fromarray(np.asarray(im)).convert("L").resize((16,16)))
print("wrist patch brightness, X=fixed attention peaks:")
for rr in range(16):
    print("".join("X" if rr*16+cc in peaks else chars[min(9,int(thumb[rr,cc])*10//256)]
                  for cc in range(16)))
print("patch RGB means at peaks:",
      {i:np.asarray(im)[(i//16)*14:(i//16+1)*14,(i%16)*14:(i%16+1)*14].mean((0,1)).round(1).tolist()
       for i in peaks})
```
