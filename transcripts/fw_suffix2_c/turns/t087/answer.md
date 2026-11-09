The replay order checks out. I’ll now test a rank-128 masked-context projector on the video text-MLP output only. Keeping the action expert and attention weights unchanged makes this a direct test of the video-mediated contamination route.
```python
def orth_columns(a, rtol=1e-9):
    u,s,vh=torch.linalg.svd(a.double(), full_matrices=False)
    return u[:,s>s[0]*rtol]

def make_oblique(contrasts, clean, fixed, rank=128, clean_energy=0.9999, min_angle=0.15):
    uc,sc,_=torch.linalg.svd(clean.double(),full_matrices=False)
    cr=int(torch.searchsorted(sc.square().cumsum(0)/sc.square().sum(),clean_energy))+1
    C=orth_columns(torch.cat([fixed.double(),uc[:,:cr]],dim=1))
    ub,sb,_=torch.linalg.svd(contrasts.double(),full_matrices=False)
    B0=ub[:,:rank]
    rem=B0-C@(C.T@B0)
    ur,sr,vhr=torch.linalg.svd(rem,full_matrices=False)
    keep=sr>=min_angle
    B=B0@vhr.T[:,keep]
    D=ur[:,keep]/sr[keep]
    return {"B":B.float(),"D":D.float(),"C":C.float(),"angles":sr.float(),
            "contrast_rank":rank,"clean_energy":clean_energy,"min_angle":min_angle}

fixed0=torch.load(W/"current_projector.pt",weights_only=False)["C"].double()
excl=torch.eye(3072,dtype=torch.float64)[:,spec["excluded"]]
fixed=torch.cat([fixed0,excl],dim=1)
clean=torch.cat([M[48:,:13].reshape(-1,3072).T,
                 U[48:].reshape(-1,3072).T],dim=1)
proj=make_oblique(dm,clean,fixed,rank=128)
torch.save(proj,W/"masked_video_projector128.pt")
name="mot.mixtures.video.text_embedding.2.weight"
orig=load_original(name)
B,D=proj["B"],proj["D"]
changed=orig-(orig@B)@D.T
save_edit(name,changed)
dest=W/"candidate_masked_video128"
dest.mkdir(exist_ok=True)
np.save(dest/(name+".npy"),changed.numpy())
print("rank",B.shape[1],"protected",proj["C"].shape[1],
      "idempotence",float((D.T@B-torch.eye(B.shape[1])).abs().max()),
      "protection_error",float((D.T@proj["C"]).abs().max()),
      "relative_weight_change",float((changed-orig).norm()/orig.norm()))
# Persistent paths for the 128-input screen.
screen_paths=[rp[i] for i in sid]+[rp[120+i] for i in sid]+rp[240:272]
(W/"screen_paths.json").write_text(json.dumps(screen_paths))
print("screen paths",len(screen_paths),screen_paths[:2],screen_paths[-2:])
with (W/"notes.md").open("a") as f:
    f.write(f"\nCandidate masked_video128: masked-prefix contrast rank128, retained{B.shape[1]}, protected{proj['C'].shape[1]}D clean/paraphrase plus unread/proprio directions. One video text_embedding.2.weight tensor; no attention gating yet. Static SVD, no fitting. Probe29 planned.\n")
```
