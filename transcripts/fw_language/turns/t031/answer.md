I will finish the schema inspection with a self-contained summary function, then send the matched controls. This is only artifact handling; the model remains unedited.
```python
def describe(x, depth=0):
    if isinstance(x, torch.Tensor):
        return {"tensor":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x, np.ndarray):
        return {"array":list(x.shape),"dtype":str(x.dtype)}
    if isinstance(x, dict):
        return {str(k):describe(v,depth+1) for k,v in list(x.items())[:25]}
    if isinstance(x, (list,tuple)):
        return {"len":len(x),"first":[describe(v,depth+1) for v in x[:2]]}
    return str(x)[:250]
print("Input schema:",json.dumps(describe(inp1),ensure_ascii=True)[:4000])
null_lights=[v for v in variants if v["category"]=="Light Conditions" and v["difficulty"] is None]
print("Unspecified difficulty lighting examples:",null_lights[:2],"count",len(null_lights))
control_eps=[]
control_map={}
for r in roll1:
    task=r["name"].split("_language_")[0]
    choices=by_task_cat[(r["suite"],task,"Light Conditions")]
    c=min(choices,key=lambda v:(v["difficulty"] if v["difficulty"] is not None else 0,v["id"]))
    key=(c["suite"],c["id"],r["seed"])
    if key not in control_map:
        control_map[key]=c
        control_eps.append({"suite":c["suite"],"id":c["id"],"seed":r["seed"]})
print("Selected controls:",[(c["suite"],c["id"],c["difficulty"]) for c in control_map.values()])
print("Request episode list:",json.dumps(control_eps,separators=(",",":")))
(W/"diagnostic_control_episodes.json").write_text(json.dumps(control_eps))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 1 baseline (backend available)\n"
            "16 original language episodes: 12 successes, 4 failures. "
            "Failures: spatial1062 (noun circumlocutions), goal1188 "
            "(wine container/cupboard), goal1246 (implicit opening prerequisite), "
            "long1208 (coffee conversational phrasing). Descriptive only. "
            "Replay inputs reside in per-episode symlink directories. "
            "No edits. 1/40 passes used.\n")
```
