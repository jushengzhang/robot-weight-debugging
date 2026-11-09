I will recover the gates as an activation-accounting check, not as trained weights, and test whether they reconstruct unseen residual updates. For the clearance feature, I will also try the conservative geometric constraint: remove the span of all source contact observations and open-gripper controls, rather than protecting only their averages.
```python
p35={r["idx"]:r for r in json.loads((W/"probe_35"/"results.json").read_text())}
p36={r["idx"]:r for r in json.loads((W/"probe_36"/"results.json").read_text())}
As=[];Ms=[];Ys=[]
for i,m in enumerate(gate_manifest):
    rz=np.load(m["resid"]); mz=np.load(m["mlp"]); az=np.load(p36[i]["activations"])
    As.append(az["calls::expert.17.attn_out::action"])
    Ms.append(mz["calls::expert.17.mlp_out::action"])
    Ys.append(rz["calls::expert.17.resid::action"]-rz["calls::expert.16.resid::action"])
A=np.array(As,dtype=np.float64); M=np.array(Ms,dtype=np.float64); Y=np.array(Ys,dtype=np.float64)
def recover_gates(A,M,Y):
    aa=(A*A).sum(0);mm=(M*M).sum(0);am=(A*M).sum(0)
    ay=(A*Y).sum(0);my=(M*Y).sum(0)
    det=aa*mm-am*am
    valid=det>1e-8*np.maximum(aa*mm,1e-20)
    den=np.where(valid,det,np.inf)
    return (ay*mm-my*am)/den,(my*aa-ay*am)/den,valid
ga,gm,valid=recover_gates(A[:4],M[:4],Y[:4])
pr=A[4:]*ga+M[4:]*gm
keep=np.ones(1024,dtype=bool);keep[[364,885]]=False
print("gate identity held-input relative error",
      round(float(np.linalg.norm((pr-Y[4:])[:,:,keep])/np.linalg.norm(Y[4:,:,keep])),4),
      "valid fraction",round(float(valid.mean()),4))
ga,gm,valid=recover_gates(A,M,Y)
print("attention gate step RMS",[round(float(v),3) for v in np.sqrt((ga[:,keep]**2).mean(1))])
print("attention gate step mean",[round(float(v),3) for v in ga[:,keep].mean(1)])
np.savez(W/"final_attention_gates.npz",attention=ga,mlp=gm,valid=valid)

gamma17=1+load_original(f"{VLM}.layers.17.input_layernorm.weight").clone()
X35={}
for stream in ["base","wrist","lang"]:
    xx=torch.stack([torch.from_numpy(np.load(p35[i]["activations"])[f"mean::vlm.16.resid::{stream}"]).float()
                    for i in range(60)])
    X35[stream]=xx/(xx.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma17
E=X35["wrist"][:44:2]; C=X35["wrist"][1:44:2]; I=X35["wrist"][44:]
sink=torch.zeros(2048);sink[674]=1
src_obs=[2*j+s for j in range(22) if j in source_indices for s in [0,1]]
centroids=torch.stack([X35["base"][src_obs].mean(0),X35["lang"][src_obs].mean(0),sink],1)
controls=torch.cat([C[source_mask].T,I.T,centroids],1)
U,S,_=torch.linalg.svd(controls,full_matrices=False)
Q=U[:,S>S[0]*1e-5]
d=E[source_mask].mean(0)
d=d-Q@(Q.T@d);d=d/d.norm()
print("protected rank",Q.shape[1],"direction support norm check",round(float((Q.T@d).norm()),6))
for label,xx in [("source_empty",E[source_mask]),("source_contact",C[source_mask]),
                 ("check_empty",E[~source_mask]),("check_contact",C[~source_mask]),
                 ("open_control",I),("base",X35["base"]),("language",X35["lang"])]:
    val=xx@d
    print(label,"mean",round(float(val.mean()),3),"std",round(float(val.std()),3),
          "range",round(float(val.min()),3),round(float(val.max()),3))
print("check paired positives",int((((E-C)@d)[~source_mask]>0).sum()),"/8")
torch.save({"direction":d,"protected_basis":Q,"source_indices":sorted(source_indices),
            "source_empty_scores":E[source_mask]@d,"check_empty_scores":E[~source_mask]@d,
            "check_contact_scores":C[~source_mask]@d},W/"late_empty_closure_direction.pt")
```
