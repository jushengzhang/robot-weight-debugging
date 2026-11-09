I’ll test a pre-GELU projection using broad and moderate-temperature context summaries, leaving out the sharpest weightings that can magnify token reweighting effects. The edit will protect a high-coverage clean/paraphrase span and the exact proprioception span, and will initially change only the video expert’s text-entry matrix.
```python
qids=[0,2,3,6,7,10,11,13]
dc=coords[:48,qids]-coords[48:96,qids]
scales=dc.square().mean((0,2)).sqrt()
print("query contrast scales",[round(float(x),6) for x in scales])
dmat=(dc/scales[None,:,None]).reshape(-1,coords.shape[-1]).T.contiguous()
uc,sc,vc=torch.linalg.svd(coords[48:].reshape(-1,coords.shape[-1]).T,full_matrices=False)
cr=int(torch.searchsorted(sc.square().cumsum(0)/sc.square().sum(),.999999))+1
P=torch.cat([load_original("proprio_encoder.weight").double(),
             load_original("proprio_encoder.bias").double()[:,None]],dim=1)
fixedc=vha@P
C=orth_columns(torch.cat([uc[:,:cr],fixedc],dim=1),rtol=1e-8)
ub,sb,vb=torch.linalg.svd(dmat,full_matrices=False)
rank=int(torch.searchsorted(sb.square().cumsum(0)/sb.square().sum(),.99))+1
print("99pct contrast rank",rank,"protected",C.shape[1])
B0=ub[:,:rank]
ur,sr,vr=torch.linalg.svd(B0-C@(C.T@B0),full_matrices=False)
keep=sr>=.15
Bc=B0@vr.T[:,keep]; Dc=ur[:,keep]/sr[keep]
Bin=(vha.T@Bc).float(); Din=(vha.T@Dc).float()
linproj={"B":Bin,"D":Din,"C_coord":C.float(),"B_coord":Bc.float(),"D_coord":Dc.float(),
         "angles":sr.float(),"rank":rank,"query_ids":qids,"protected_rank":C.shape[1]}
torch.save(linproj,W/"linear_context_projector.pt")
n="mot.mixtures.video.text_embedding.0.weight"
o=load_original(n)
new=o-(o@Bin)@Din.T
save_edit(n,new)
destlin=W/"candidate_linear_video"; destlin.mkdir(exist_ok=True)
np.save(destlin/(n+".npy"),new.numpy())
print("retained rank",Bin.shape[1],"relative weight change",float((new-o).norm()/o.norm()),
      "proprio residual",float((Din.double().T@P).norm()/P.norm()))
with (W/"notes.md").open("a") as f:
    f.write(f"\nCandidate linear_context_video: one video text_embedding.0.weight; original bias and all later layers untouched. Broad/moderate queries{qids}, 99% contrast rank{rank}, retained{Bin.shape[1]}, protects{C.shape[1]} clean/proprio directions. Clean uncentered energy .999999. Static observable-context SVD projector; probe34 planned.\n")
```
