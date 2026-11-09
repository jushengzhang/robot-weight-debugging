The dense clean scan supports a much better protection basis: 512 directions preserve about 99.9% of the measured clean variation, while roughly 45% of matched tagged/clean contrast energy remains outside it. I’ll test a rank-16 oblique projection built from per-instruction matched contrasts. It changes only the text-entry matrix of each expert and exactly protects the proprioception span.
```python
Q=torch.linalg.qr(torch.cat([Qp,Vhc[:512].T],dim=1),mode="reduced").Q
Dmeans=torch.stack([D[torch.tensor([p[0]==i for p in pairs])].mean(0) for i in range(24)])
_,Sm,Vhm=torch.linalg.svd(Dmeans,full_matrices=False)
B=Vhm[:16].T.contiguous()
Bo=B-Q@(Q.T@B)
G=Bo.T@B
U=Bo@torch.linalg.inv(G)
print("Oblique projector check U^T B error:",float((U.T@B-torch.eye(16)).abs().max()))
print("Protection check:",float((Q.T@U).abs().max()),
      "detector norm range:",[float(U.norm(dim=0).min()),float(U.norm(dim=0).max())])
print("Matched contrast remaining:",round(float((D-(D@U)@B.T).norm()/D.norm()),4))
print("Clean measured relative change:",round(float(((C@U)@B.T).norm()/C.norm()),6))
torch.save(dict(B=B,U=U,Q=Q,Dmeans=Dmeans,Sm=Sm,Vhm=Vhm,
                rank=16,source="per_instruction_mutual_prefix_matches"),W/"dense_prefix_projector.pt")
def install_projector(B,U,strength=1.,experts=("action","video")):
    clear_edits()
    result={}
    for ex in experts:
        n=f"mot.mixtures.{ex}.text_embedding.0.weight"
        w=load_original(n)
        delta=((w@B)@U.T)*strength
        save_edit(n,w-delta)
        result[ex]=float(delta.norm()/w.norm())
    return result
print("Installed rank16 both-expert edit:",install_projector(B,U))

helper=r'''
import numpy as np, json
from pathlib import Path
def probe_metrics(k, idx=None):
    root=Path(WORK)
    if idx is None: idx=list(range(56))
    base=np.stack([np.load(root/"probe_3"/f"probe_orig_{i}_action.npy") for i in range(56)])
    arr=np.stack([np.load(root/f"probe_{k}"/f"probe_edit_{j}_action.npy") for j in range(len(idx))])
    lookup={i:arr[j] for j,i in enumerate(idx)}
    rs=[i for i in range(24) if i in lookup and i+24 in lookup]
    ls=[i for i in range(48,56) if i in lookup]
    rms=lambda a:float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2)))
    er=np.stack([lookup[i] for i in rs]); ec=np.stack([lookup[i+24] for i in rs])
    br=base[rs];bc=base[[i+24 for i in rs]]
    gap=rms(br-bc)
    z=dict(passno=k,baseline_gap=gap,gap=rms(er-bc),ratio=rms(er-bc)/gap,
           clean_drift=rms(ec-bc),language_drift=rms(np.stack([lookup[i]-base[i] for i in ls])) if ls else None,
           raw_change=rms(er-br),correction_fraction=float(np.sum((er-br)*(bc-br))/np.sum((bc-br)**2)),
           edited_gap=rms(er-ec))
    meta=json.loads((root/"probe_items.json").read_text())
    z["category_ratio"]={}
    for cat in sorted(set(meta[i]["category"] for i in rs)):
        jj=[j for j,i in enumerate(rs) if meta[i]["category"]==cat]
        z["category_ratio"][cat]=round(rms(er[jj]-bc[jj])/rms(br[jj]-bc[jj]),4)
    print(json.dumps(z))
    with (root/"notes.md").open("a") as f:
        f.write("\nProbe metrics: "+json.dumps(z)+"\n")
    return z
'''
(W/"analysis_helpers.py").write_text(helper)
exec(helper)
with (W/"notes.md").open("a") as f:
    f.write("\n### Candidate dense prefix-mean rank16\n182 mutual raw/clean context matches across24 instructions; per-instruction contrast means define16D oblique projector. "
            "Protect512 dense clean/paraphrase PCs plus exact9D proprio span. Only two text_embedding.0.weight tensors changed. Probe19 pending.\n")
```
