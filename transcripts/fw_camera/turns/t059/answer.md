The final camera-sensitive directions are not yet stable enough across task subsets to justify a readout edit. I will check how many additional matched practice states are available, so that any such edit is based on repeatable structure rather than a small-sample direction.
```python
pv_obj = json.loads((Path(BENCH)/"practice_variants.json").read_text())
print("Practice type:",type(pv_obj).__name__)
if isinstance(pv_obj,dict):
    print("keys:",list(pv_obj)[:8])
    pv = pv_obj.get("variants",pv_obj.get("practice",[]))
else:
    pv = pv_obj
print("count",len(pv),"example",str(pv[:1])[:600])
from collections import Counter, defaultdict
pmap = {(v["suite"],int(v["id"])):v for v in pv}
discovery_keys = {(v["suite"],int(v["id"])) for v in part["discovery"]}
validation_keys = {(v["suite"],int(v["id"])) for v in part["validation"]}
print("All categories",dict(Counter(v["category"] for v in pv)))
print("Discovery categories",dict(Counter(v["category"] for v in pv if (v["suite"],int(v["id"])) in discovery_keys)))
cam_dis = [v for v in pv if v["category"]=="Camera Viewpoints" and (v["suite"],int(v["id"])) in discovery_keys]
cam_groups = defaultdict(list)
for v in cam_dis:
    task=v["name"].split("_view_")[0]
    cam_groups[(v["suite"],task)].append(v)
print("Discovery camera tasks",len(cam_groups),"per-task counts",dict(Counter(len(vs) for vs in cam_groups.values())))
print("Camera example names",[(v["difficulty"],v["name"]) for v in cam_dis[:5]])
```
