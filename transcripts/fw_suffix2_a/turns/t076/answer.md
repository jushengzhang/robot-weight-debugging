Video-only gating also gains +5.2 points, with seven rescues and two regressions. That provides causal evidence that the tag can affect actions through the video expert. I’ll now isolate the action expert’s direct text-attention route.
```python
restore_script=r'''
import numpy as np, torch
from pathlib import Path
def restore_dense_candidate(experts=("action","video"),dose=1.,threshold=.20):
    for n in list(edited()):
        drop_edit(n)
    src=Path(WORK)/"candidate_dense_gate_t020_both"
    meta=torch.load(src/"metadata.pt",weights_only=False)
    for fp in src.glob("*.npy"):
        name=fp.name[:-4]
        if not any(f".{e}." in name for e in experts):continue
        x=torch.from_numpy(np.load(fp))
        if name.endswith("cross_attn.q.bias") and dose!=1.:
            old=load_original(name)
            mask=x!=old
            x[mask]*=dose
        if name.endswith("text_embedding.0.bias") and threshold!=.20:
            ex="action" if ".action." in name else "video"
            x[meta["specs"][ex]["hidden"]]=-meta["gain"]*threshold
        save_edit(name,x)
    return len(edited())
def paired_rollout(passnos):
    root=Path(WORK)
    if isinstance(passnos,int):passnos=[passnos]
    rows=[]
    for k in passnos:
        data=[__import__("json").loads(l) for l in (root/f"roll_{k}.jsonl").read_text().splitlines()]
        by={}
        for x in data:
            key=(k,x["suite"],x["id"],x["seed"])
            by.setdefault(key,{})[x["model"]]=x
        rows.extend((p["original"],p["edited"]) for p in by.values() if "original" in p and "edited" in p)
    d=np.array([int(e["success"])-int(o["success"]) for o,e in rows],float)
    out=dict(passes=passnos,n=len(d),original_success=sum(o["success"] for o,e in rows),
             edited_success=sum(e["success"] for o,e in rows),wrong_to_right=int((d==1).sum()),
             right_to_wrong=int((d==-1).sum()),diff_pp=100*float(d.mean()),
             se_pp=100*float(d.std(ddof=1)/np.sqrt(len(d))))
    out["ci95_pp"]=[out["diff_pp"]-1.96*out["se_pp"],out["diff_pp"]+1.96*out["se_pp"]]
    return out
'''
(W/"candidate_helpers.py").write_text(restore_script)
exec(restore_script)
print("Pass22:",json.dumps(paired_rollout(22)))
print("Installed action-only tensors:",restore_dense_candidate(experts=("action",)))
with (W/"notes.md").open("a") as f:
    f.write("\n### Pass22 video-only causal test\nSame96 development episodes:47->52 successes,+5.21pp,SE3.1pp,7 gains/2 losses. "
            "Indirect video route alone reproduces total gain with fewer regressions. Action-only now installed for pass23. "
            "candidate_helpers.py restores frozen gate, changes only expert selection/dose/threshold, and computes paired statistics.\n")
```
