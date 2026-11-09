The pre-GELU edit also failed. I’ll now separate two operations that the best earlier projector changed together: text keys choose what to attend to, while text values supply the content. I’ll apply the earlier weighted-context correction only to video cross-attention values, leaving keys, text embeddings, and the action expert original.
```python
clear_edits()
hp=torch.load(W/"weighted_video_projector.pt",weights_only=False)
T=load_original("mot.mixtures.video.text_embedding.2.weight").double()
bias=load_original("mot.mixtures.video.text_embedding.2.bias").double()
By0=orth_columns(T@hp["B"].double(),rtol=1e-8)
Cy=orth_columns(torch.cat([T@hp["C"].double(),bias[:,None]],dim=1),rtol=1e-8)
uy,sy,vhy=torch.linalg.svd(By0-Cy@(Cy.T@By0),full_matrices=False)
ky=sy>=.15
By=(By0@vhy.T[:,ky]).float()
Dy=(uy[:,ky]/sy[ky]).float()
vp={"B":By,"D":Dy,"C":Cy.float(),"angles":sy.float(),
    "source":"weighted_video_projector","route":"value","dose":1.0}
torch.save(vp,W/"weighted_video_output_projector.pt")
print("output projector rank",By.shape[1],"protected",Cy.shape[1],
      "bias leakage",float((Dy.double().T@bias).norm()))
stats=[]
for layer in range(30):
    n=f"mot.mixtures.video.blocks.{layer}.cross_attn.v.weight"
    o=load_original(n)
    a=o-(o@By)@Dy.T
    stats.append(float((a-o).norm()/o.norm()))
    save_edit(n,a)
print("active tensors",len(edited()),"relative change min/mean/max",
      min(stats),float(np.mean(stats)),max(stats))
with (W/"notes.md").open("a") as f:
    f.write(f"\nPass34 pre-GELU candidate rejected: gap1.3735, arm10 1.3052, clean/lang drift.00503/.01237. All edits cleared. Pass35 candidate: video-only cross_attn.v.weight in all30 blocks. Output-space image of previously successful weighted118D hidden contrast span, protected clean/proprio/bias span, retained{By.shape[1]}; values only, keys/text/action original. Static oblique projector, no fitting.\n")
```
