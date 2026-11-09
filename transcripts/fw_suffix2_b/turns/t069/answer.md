The two text-input matrices now match the intended half-dose construction. I will finish the integrity check and save reusable analysis helpers so another kernel restart will not lose the measurement definitions or candidate metadata.
```python
import hashlib
def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()
mismatch=[]
for n in gate_names:
    if n.endswith("text_embedding.0.weight"):
        continue
    if file_hash(W/"edited"/(n+".npy"))!=file_hash(snap/(n+".npy")):
        mismatch.append(n)
print("NON-PROJECTION GATE MISMATCHES",mismatch)
assert not mismatch and len(edited())==308
helper_source = r'''
from pathlib import Path
import json, numpy as np, torch, shutil
W=Path(WORK)
sp=json.loads((W/"splits.json").read_text())
proj=torch.load(W/"rich_context_projection.pt",weights_only=False,map_location="cpu")
def actions_for(k,field="action"):
    rows=sorted(json.loads((W/f"probe_{k}/results.json").read_text()),key=lambda x:x["idx"])
    return np.stack([np.load(r[field]) for r in rows])
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
A0=actions_for(4)
An0=actions_for(4,"action_norm")
def probe_metrics(k,record=True):
    a=actions_for(k)
    gap=A0[:24]-A0[24:48]; shift=a[:24]-A0[:24]
    met={"pass_id":k,"baseline_gap":rms(gap),"raw_to_original_clean":rms(a[:24]-A0[24:48]),
         "gap_ratio":rms(a[:24]-A0[24:48])/rms(gap),
         "edited_raw_clean_gap":rms(a[:24]-a[24:48]),
         "clean_drift":rms(a[24:48]-A0[24:48]),"language_drift":rms(a[48:]-A0[48:]),
         "raw_change":rms(shift),"correction_fraction":float(np.sum(shift*(-gap))/np.sum(gap**2)),
         "correction_cosine":float(np.sum(shift*(-gap))/np.sqrt(np.sum(shift**2)*np.sum(gap**2)))}
    cats=[v["category"] for v in sp["diagnostic"]]; met["by_category"]={}
    for cat in sorted(set(cats)):
        ix=np.array([i for i,c in enumerate(cats) if c==cat])
        met["by_category"][cat]={"gap_ratio":rms(a[ix]-A0[ix+24])/rms(gap[ix]),
            "clean_drift":rms(a[ix+24]-A0[ix+24]),"raw_change":rms(shift[ix])}
    if record:
        (W/f"probe_{k}_metrics.json").write_text(json.dumps(met,indent=2))
        with (W/"notes.md").open("a") as f:f.write("\nProbe metrics: "+json.dumps(met)+"\n")
    return met
def paired_rolls(ks):
    pairs=[]
    for k in ks:
        rows=[json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines()]
        by={}
        for r in rows:by.setdefault((r["suite"],r["id"],r["seed"]),{})[r["model"]]=r
        pairs.extend(z for z in by.values() if "original" in z and "edited" in z)
    def summary(pp):
        ds=np.array([int(p["edited"]["success"])-int(p["original"]["success"]) for p in pp],dtype=float)
        return {"n":len(pp),"original":sum(p["original"]["success"] for p in pp),
                "edited":sum(p["edited"]["success"] for p in pp),
                "gains":int(np.sum(ds==1)),"losses":int(np.sum(ds==-1)),
                "diff_pp":float(ds.mean()*100),"se_pp":float(ds.std(ddof=1)/np.sqrt(len(ds))*100)}
    ans=summary(pairs)
    ans["by_category"]={c:summary([p for p in pairs if p["original"]["category"]==c])
        for c in sorted(set(p["original"]["category"] for p in pairs))}
    return ans
def snapshot_candidate(label):
    dest=W/"candidates"/label;dest.mkdir(parents=True,exist_ok=True)
    for n in edited():shutil.copyfile(W/"edited"/(n+".npy"),dest/(n+".npy"))
    shutil.copyfile(W/"current_candidate.json",dest/"candidate.json")
    return str(dest)
def update_hybrid_projection(rank=256,dose=1.,experts=("action","video")):
    snap=W/"candidates/hybrid_r256_d1_t05_g128"
    u=proj["U"][:,:rank].double(); b=proj["B_nearest"][:,:rank].double()
    for ex,h in [("action",296),("video",2844)]:
        n=f"mot.mixtures.{ex}.text_embedding.0.weight"
        o=load_original(n).double()
        d=dose if ex in experts else 0.
        a=(o-d*(o@b)@u.T).float()
        a[h]=torch.from_numpy(np.load(snap/(n+".npy")))[h]
        save_edit(n,a)
    cfg={"kind":"hybrid_oblique_suffix_gate","rank":rank,"dose":dose,"threshold":.5,
         "gain":128.,"strength":6.,"experts":list(experts)}
    (W/"current_candidate.json").write_text(json.dumps(cfg,indent=2))
    return cfg
'''
(W/"analysis_helpers.py").write_text(helper_source)
exec(helper_source)
print("HELPERS SAVED; CANDIDATE",json.loads((W/"current_candidate.json").read_text()))
```
