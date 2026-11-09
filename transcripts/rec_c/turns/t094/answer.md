There are 22 scenes with both stable empty-closure and contact-width observations, though some pairs also differ in arm pose. I will probe those images and reserve eight scenes to check whether any proposed visual direction generalizes before making an edit. If it only separates the source pairs, I will discard it.
```python
rng=np.random.default_rng(3201)
order=rng.permutation(len(pairs)).tolist()
source_indices=set(order[:14])
contact_manifest=[]
cpdir=W/"contact_probe"; cpdir.mkdir(exist_ok=True)
for j,p in enumerate(pairs):
    for cls in ["empty","contact"]:
        idx=len(contact_manifest)
        target=Path(p[cls]["input"]).resolve()
        link=cpdir/f"{idx:02d}.pt"
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(target)
        contact_manifest.append({"index":idx,"id":p["id"],"class":cls,
                                 "split":"source" if j in source_indices else "check",
                                 "input":str(link),"source":str(target),
                                 "distance":p["distance"]})
(W/"contact_manifest.json").write_text(json.dumps(contact_manifest))
print("source scenes",[p["id"] for j,p in enumerate(pairs) if j in source_indices])
print("check scenes",[p["id"] for j,p in enumerate(pairs) if j not in source_indices])
print("paths", [f"WORK/contact_probe/{j:02d}.pt" for j in range(len(contact_manifest))])
print("active edits",edited())
```
