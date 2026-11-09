I will compare rank eight with rank one and inspect how the contrast direction is transmitted through the video context projection. If removing more directions does not help, the next useful distinction is whether the error comes from attention keys, which choose tokens, or values, which carry their content.
```python
m12=screen_metrics(12)
shutil.copyfile(W/"current_projector.pt",W/"projector_P8.pt")
sd=W/"candidate_video_hidden_P8"; sd.mkdir(exist_ok=True)
for n in edited(): np.save(sd/(n+".npy"),load(n).numpy())
# Weight-only conditioning analysis of video context projection.
s_w2=torch.linalg.svdvals(v_w2.double())
print("VIDEO TEXT OUTPUT SINGULAR VALUES:",[round(float(s_w2[i]),6) for i in (0,100,1000,2000,2900,3000,3060,3071)])
print("condition",float(s_w2[0]/s_w2[-1]))
# Contrast after the text output projection, using measured feature means.
Yc=cleanF@v_w2.double().T
Yu=diffU@v_w2.double().T
Ym=diffM@v_w2.double().T
for lab,D in (("context_uniform",Yu),("context_masked",Ym)):
    _,s,_=torch.linalg.svd(D,full_matrices=False)
    print(lab,"norm",float(D.norm(dim=1).mean()),"energy1/4/8",
          [round(float((s[:k]**2).sum()/(s*s).sum()),4) for k in (1,4,8)])
```
