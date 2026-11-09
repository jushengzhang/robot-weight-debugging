The weight scan found unusually high paired query/key gains in a few late-layer heads, especially layer 29 heads 4, 7, and 8. High gains can make attention brittle, but I will check their channel structure before drawing conclusions. I will also expand the diagnostic panel to all four suites using the least-severe available lighting control for each task; lighting remains a control, not an exact unperturbed match.```python
print("Head gain correlations (weight-only):")
for a,b in [("action_q_gain","video_k_gain"),
            ("action_q_gain","action_k_gain"),
            ("video_q_gain","video_k_gain")]:
    print(a,b, round(float(np.corrcoef(hs[a].ravel(),hs[b].ravel())[0,1]),5))
print("Layer: median/max action-q gain; median/max video-k gain; max reader head")
for l in range(30):
    print("%02d %.3f %.3f %.3f %.3f H%02d" % (
        l,np.median(hs["action_q_gain"][l]),hs["action_q_gain"][l].max(),
        np.median(hs["video_k_gain"][l]),hs["video_k_gain"][l].max(),
        int(reader_proxy[l].argmax())))
print("\nDetailed head norm gains:")
for l,h in [(29,4),(29,7),(29,8),(18,10),(25,3),(5,4),(12,21),(0,0)]:
    details = {}
    for ex, qk in [("action","q"),("action","k"),("video","q"),("video","k")]:
        g=load_original(nm(ex,l,f"self_attn.norm_{qk}.weight")).reshape(24,128)[h]
        details[ex+"_"+qk] = [round(float(x),4) for x in (g.mean(),g.std(),g.min(),g.max())]
    print(l,h,details)

diag_variants, diag_groups = [], []
rng = np.random.default_rng(1701)
for suite in sorted(set(x["suite"] for x in practice)):
    eligible = []
    for (s, task), xs in by_task.items():
        if s != suite:
            continue
        lo=[x for x in xs if x["category"]=="Camera Viewpoints" and x["difficulty"] in (1,2)]
        hi=[x for x in xs if x["category"]=="Camera Viewpoints" and x["difficulty"] in (4,5)]
        ctrl=[x for x in xs if x["category"]=="Light Conditions"]
        if lo and hi and ctrl:
            eligible.append((task,lo,hi,ctrl))
    eligible.sort(key=lambda z:z[0])
    print(suite, "eligible tasks:",len(eligible))
    for j in rng.permutation(len(eligible))[:8]:
        task,lo,hi,ctrl=eligible[j]
        chosen=[lo[int(rng.integers(len(lo)))],hi[int(rng.integers(len(hi)))],
                min(ctrl,key=lambda x:(x["difficulty"] if x["difficulty"] is not None else 9,x["id"]))]
        seed=1100+len(diag_groups)
        diag_groups.append({"suite":suite,"task":task,"seed":seed,"ids":[x["id"] for x in chosen]})
        diag_variants.extend([{**x,"seed":seed} for x in chosen])
diag_episodes=[{k:x[k] for k in ("suite","id","seed")} for x in diag_variants]
(W/"diagnostic_panel.json").write_text(json.dumps(diag_variants,indent=2))
(W/"diagnostic_groups.json").write_text(json.dumps(diag_groups,indent=2))
print("Diagnostic panel:",len(diag_episodes),"episodes",len(diag_groups),"tasks")
print(json.dumps(diag_episodes,separators=(",",":")))
```
