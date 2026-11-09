Environment perturbations have a much larger uniform-color shift than the other perturbations, and 99.8% of that shift lies along one nearly achromatic direction. I will halve the patch embedding’s response to only that spatially uniform color component, compensating its bias at the canonical camera mean. This preserves edges, other color directions, and the vast majority of the filter weights.
```python
alpha=0.5
mu_v=float(14*np.dot(color_u,canonical_rgb))
dc_out=Pflat@color_v
newW=(Pflat-alpha*dc_out[:,None]*color_v[None,:]).reshape_as(PW)
newB=load_original(patchb)+alpha*dc_out*mu_v
save_edit(patchw,newW)
save_edit(patchb,newB)
torch.save({"v":color_v,"canonical_mean_projection":mu_v,
            "rgb_direction":color_u,"canonical_rgb":canonical_rgb},
           W/"subspaces/patch_dc_direction.pt")
dst=W/"candidate_patch_dc_half"
dst.mkdir(exist_ok=True)
for p in (W/"edited").iterdir():
    if p.is_file(): shutil.copy2(p,dst/p.name)
# Stratified diagnostic evaluation, chosen without reference to outcomes.
splits=json.loads((W/"split_manifest.json").read_text())
diagset=set(splits["diagnostic"])
def hkey(e,tag):
    return hashlib.sha256((tag+":"+str(e["id"])).encode()).hexdigest()
env_by_task={}
for e in eps:
    if e["id"] in diagset and e["ptype"]=="environment":
        env_by_task.setdefault((e["suite"],e["task"]),[]).append(e)
dc_ids=[]
for key,es in sorted(env_by_task.items()):
    dc_ids += [e["id"] for e in sorted(es,key=lambda e:hkey(e,"patch-dc-diag"))[:2]]
for pt in ["object","semantic"]:
    for suite in sorted({e["suite"] for e in eps}):
        es=[e for e in eps if e["id"] in diagset and e["ptype"]==pt and e["suite"]==suite]
        dc_ids += [e["id"] for e in sorted(es,key=lambda e:hkey(e,"patch-dc-control"))[:6]]
assert len(dc_ids)==128 and len(set(dc_ids))==128
(W/"patch_dc_test_ids.json").write_text(json.dumps(dc_ids))
print("Patch weight relative change",round(float((newW-PW).norm()/PW.norm()),4))
print("Test composition",{pt:sum(byid[i]["ptype"]==pt for i in dc_ids) for pt in ["environment","object","semantic"]})
print("Test IDs:",dc_ids)
with (W/"notes.md").open("a") as f:
    f.write("\n## Environment input diagnosis and DC-filter candidate\n"
            "Initial camera mean RGB shifts vs semantic controls: environment norm.643, "
            "object.039, position.033, task.005. Environment RGB shift rank1 energy99.8%; "
            "direction[.543,.599,.588], nearly luminance. "
            "Patch convolution DC direction accounts for only0.59% of weight energy. "
            "Candidate W'=W-.5(Wv)v^T, where v is spatially uniform14x14 with this RGB direction; "
            "b'=b+.5(Wv)(v^T canonical camera mean). "
            "Preserves orthogonal color/edge filters and response at canonical mean. "
            "New stratified diagnostic set:80environment (2/task),24object,24semantic; seed17.\n")
```
