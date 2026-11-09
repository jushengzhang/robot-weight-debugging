Some severe loops did resolve under the half-strength edit, including scene 979, but several previously successful episodes became loops. I will test full removal of the same visual direction as a dose-response experiment, without changing any other pathway.
```python
import random
ids = [r["id"] for r in practice_raw]
dset = set(splits["discovery"])
for method, subset in [
    ("python-random",random.Random(7).sample(ids,64)),
    ("numpy-default",np.random.default_rng(7).choice(ids,64,replace=False).tolist()),
    ("numpy-legacy",np.random.RandomState(7).choice(ids,64,replace=False).tolist())]:
    print("seed7 sample",method,"overlap",len(set(subset)&dset),"/63")
u = torch.load(W/"visual_grip_direction.pt",weights_only=False)["direction"].float()
u = u/u.norm()
proj = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
lnbias_name = next(n for n in NAMES if n.endswith("vision_model.post_layernorm.bias"))
pw = load_original(proj+"weight")
pb = load_original(proj+"bias")
lnb = load_original(lnbias_name)
alpha=1.0
pu = pw@u
save_edit(proj+"weight",pw-alpha*torch.outer(pu,u))
save_edit(proj+"bias",pb+alpha*pu*(u@lnb))
def backup_edits(tag):
    p=W/tag
    p.mkdir(exist_ok=True)
    manifest={}
    for i,n in enumerate(edited()):
        fn=f"tensor_{i:03d}.pt"
        torch.save(load(n),p/fn)
        manifest[n]=fn
    (p/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return str(p)
print("saved",backup_edits("candidate_visual_grip_full"))
with (W/"notes.md").open("a") as f:
    f.write("\nNext: alpha1.0 visual aperture projection for dose-response; no other edits.\n")
del pw,pb,lnb,pu
```
