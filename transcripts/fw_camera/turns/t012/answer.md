I will prepare reversible edit utilities and preselect the discovery rollout panel now. This avoids choosing evaluation episodes after seeing an edit's outcomes, and ensures every candidate can be removed or restored exactly. No weights will be changed in this step.```python
import shutil

def clear_edits():
    for name in list(edited()):
        drop_edit(name)

def snapshot_edits(label, metadata=None):
    dest=W/"candidates"/label
    dest.mkdir(parents=True,exist_ok=True)
    manifest={}
    for i,name in enumerate(sorted(edited())):
        fn=f"tensor_{i:03d}.pt"
        torch.save(load(name),dest/fn)
        manifest[name]=fn
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (dest/"metadata.json").write_text(json.dumps(metadata or {},indent=2))
    return str(dest)

def restore_edits(label):
    dest=W/"candidates"/label
    manifest=json.loads((dest/"manifest.json").read_text())
    clear_edits()
    for name,fn in manifest.items():
        t=torch.load(dest/fn,map_location="cpu",weights_only=True)
        assert tuple(t.shape)==tuple(SHAPES[name]) and torch.isfinite(t).all()
        save_edit(name,t)
    return list(edited())

def edit_summary():
    rows=[]
    for name in sorted(edited()):
        a=load_original(name)
        b=load(name)
        d=b-a
        rows.append({"name":name,"changed":int((d!=0).sum()),
                     "total":a.numel(),"relative_frobenius":float(d.norm()/a.norm().clamp_min(1e-12)),
                     "max_abs_change":float(d.abs().max())})
    return rows

def attenuate_output_heads(heads, dose):
    # Local head-column ablation, always relative to original weights.
    clear_edits()
    for l in sorted(set(l for l,h in heads)):
        name=nm("action",l,"self_attn.o.weight")
        t=load_original(name).clone()
        for ll,h in heads:
            if ll==l:
                t[:,128*h:128*(h+1)] *= 1.0-dose
        save_edit(name,t)

def attenuate_norm_coordinates(head_coordinates, dose, cap=2.0):
    # Only explicitly selected excessive coordinates are altered.
    clear_edits()
    for l, selected in head_coordinates.items():
        name=nm("action",l,"self_attn.norm_q.weight")
        t=load_original(name).clone().reshape(24,128)
        for h,coords in selected.items():
            old=t[h,coords].clone()
            target=old.sign()*old.abs().clamp(max=cap)
            t[h,coords]=old+dose*(target-old)
        save_edit(name,t.reshape(-1))

def balanced_pick(pool,n,seed):
    rng=np.random.default_rng(seed)
    groups=collections.defaultdict(list)
    for x in pool:
        groups[(x["suite"],x["difficulty"])].append(x)
    keys=sorted(groups,key=str)
    queues={k:[groups[k][i] for i in rng.permutation(len(groups[k]))] for k in keys}
    out=[]
    while len(out)<n and any(queues.values()):
        for j in rng.permutation(len(keys)):
            if queues[keys[j]]:
                out.append(queues[keys[j]].pop())
                if len(out)==n:
                    break
    return out

screen_cam=balanced_pick([x for x in discovery if x["category"]=="Camera Viewpoints"],128,2401)
screen_cam_episodes=[{"suite":x["suite"],"id":x["id"],"seed":2100+i} for i,x in enumerate(screen_cam)]
(W/"screen_camera_variants.json").write_text(json.dumps(screen_cam,indent=2))
(W/"screen_camera_episodes.json").write_text(json.dumps(screen_cam_episodes,indent=2))
print("Discovery camera panel:",
      dict(collections.Counter(x["suite"] for x in screen_cam)),
      dict(collections.Counter(x["difficulty"] for x in screen_cam)))
print("Validation camera variants remain:",sum(x["category"]=="Camera Viewpoints" for x in validation))
print("Edit registry prepared; actual edits:",list(edited()))
```
