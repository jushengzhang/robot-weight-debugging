The larger diagnostic set shows 17/65 background failures versus 2/37 lighting failures, with the worst results at difficulty 5 and in long-horizon tasks. I will now compare matched-task attention shifts and inspect action traces. I will use first-observation probes next so that failed trajectories do not masquerade as the cause of failure.

```python
roll2=[json.loads(x) for x in (W/"roll_2.jsonl").read_text().splitlines() if x.strip()]
r2map={(r["suite"],r["id"]):r for r in roll2}
act2={}
for r in roll2:
    with np.load(r["activations"],allow_pickle=False) as z:
        act2[(r["suite"],r["id"])]=dict(z)
keys2=list(next(iter(act2.values())))
print("Activation keys/shapes sample:",[(k,np.asarray(next(iter(act2.values()))[k]).shape) for k in keys2[:8]])
print("BG failures:")
for r in roll2:
    if r["category"]=="Background Textures" and not r["success"]:
        a=np.load(r["actions_npy"],allow_pickle=False)
        aa=a.reshape(-1,a.shape[-1])
        print(r["suite"],r["id"],"diff",r["difficulty"],"steps",r["steps"],
              "actions_shape",list(a.shape),"mean_xyz",np.round(aa[:,:3].mean(0),3).tolist(),
              "grip_pos_frac",round(float((aa[:,-1]>0).mean()),3),
              "task",bykey[(r["suite"],r["id"])]["_task"][:65])

def paired_layer_contrasts(which="light",what="attn_out",expert="action",camera="action"):
    ps=[p for p in pairs_meta if p.get(which) is not None
        and (p["suite"],p["high"]) in act2 and (p["suite"],p[which]) in act2]
    rows=[]
    for l in range(30):
        k=f"mean::{expert}.{l}.{what}::{camera}"
        if k not in keys2:
            continue
        X=np.stack([act2[(p["suite"],p["high"])][k] for p in ps])
        C=np.stack([act2[(p["suite"],p[which])][k] for p in ps])
        D=X-C
        norm=np.linalg.norm(D,axis=1)
        base=np.linalg.norm(C,axis=1)
        fail=np.array([not r2map[(p["suite"],p["high"])]["success"] for p in ps])
        ss=np.linalg.svd(D,compute_uv=False)
        rows.append((l,float(np.median(norm/np.maximum(base,1e-8))),
                     float(norm[fail].mean()) if fail.any() else 0,
                     float(norm[~fail].mean()) if (~fail).any() else 0,
                     float(ss[0]**2/(ss**2).sum())))
    return rows
for expert,cam in [("action","action"),("video","agent"),("video","wrist")]:
    rows=paired_layer_contrasts("low",expert=expert,camera=cam)
    print("\nMatched high-low episode-average",expert,cam,
          "(L, relative delta, failed delta, successful delta, top1 fraction):")
    print([(l,round(a,3),round(b,3),round(c,3),round(d,3)) for l,a,b,c,d in rows])

input_example=W/"roll_2_frames"/"libero_10_v15_s17_original"/"t0000_inputs.pt"
inp=torch.load(input_example,map_location="cpu",weights_only=False)
print("\nPolicy input structure:",type(inp).__name__)
if isinstance(inp,dict):
    print({k:(list(v.shape) if hasattr(v,"shape") else type(v).__name__) for k,v in inp.items()})
else:
    print(str(inp)[:500])
note("## Pass 2: full original diagnostic\n"
     "- 102 episodes: BG 48/65 (73.85%), light 35/37 (94.59%).\n"
     "- Total 19 failures; 17 BG, 2 light. Difficulty 5 overall 4/11; long suite 13/26.\n"
     "- Attention outputs recorded in all 30 layers for both experts; first observations retained for deconfounded probes.\n"
     "- Passes used: 2/40; no edits.")
```
