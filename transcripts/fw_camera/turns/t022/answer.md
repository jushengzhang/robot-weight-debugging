I will check the practice camera variants' parameter families and the coverage of our fixed panels. A repair that works only for one kind of camera movement would be weak evidence of robustness, so the discovery and reserved validation sets should cover the available viewpoint changes. I will also record the backend delay explicitly rather than treating refused requests as passes.```python
def view_parameters(x):
    m=re.search(r"_view_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_initstate_",x["name"])
    if m:
        try:
            return tuple(float(z) for z in m.groups())
        except ValueError:
            pass
    return None

def view_family(x):
    p=view_parameters(x)
    if p is None:
        return "unparsed"
    base=(0.,0.,100.,0.,0.)
    return "changed_"+("_".join(str(i) for i,(v,b) in enumerate(zip(p,base)) if v!=b) or "none")

for label,pool in [
    ("all practice cameras",[x for x in practice if x["category"]=="Camera Viewpoints"]),
    ("diagnostic cameras",[x for x in diag_variants if x["category"]=="Camera Viewpoints"]),
    ("discovery screen",screen_cam),
    ("reserved validation",[x for x in validation if x["category"]=="Camera Viewpoints"])]:
    print(label,dict(collections.Counter(view_family(x) for x in pool)))
print("Camera parameter ranges in practice only:")
params=np.array([view_parameters(x) for x in practice
                 if x["category"]=="Camera Viewpoints" and view_parameters(x) is not None])
print("min:",params.min(0).tolist(),"max:",params.max(0).tolist())
print("Diagnostic task, mild/high parameter tuples:")
for g in diag_groups[:8]:
    xs=[x for x in diag_variants if x["suite"]==g["suite"] and x["seed"]==g["seed"]
        and x["category"]=="Camera Viewpoints"]
    print(g["suite"],g["seed"],[view_parameters(x) for x in xs])
elapsed=time.time()-(W/"partition.json").stat().st_mtime
print("Wall time since partition creation (minutes):",round(elapsed/60,2))
with open(W/"notes.md","a") as f:
    f.write("\n## Backend status\n"
            "- All rollout requests so far were refused as offline, at no pass cost.\n"
            "- Forward passes still 0/40; no observations, outcomes, or activation evidence yet.\n"
            "- Current best remains original; no weights changed based on weight-only hypotheses.\n")
print("Prepared artifacts:",len(list(W.glob("*.json"))),"JSON files.")
```
