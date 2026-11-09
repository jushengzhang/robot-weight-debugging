I will summarize the opposite-sign probe and verify that the remaining validation scenes are disjoint from discovery and the first validation panel. Before seeing their outcomes, I will also make the final acceptance rule explicit: a gain above two standard errors, fewer loop failures, and at most two percentage points of baseline-success regressions on this recovery panel.
```python
r47=json.loads((W/"probe_47"/"results.json").read_text())
s47=[]
for i in range(16):
    pair=[next(r for r in r47 if r["idx"]==i and r["arm"]==a) for a in ["orig","edit"]]
    a,b=[np.load(r["action"]) for r in pair]
    s47.append({"i":i,"group":"empty" if i<4 else "holding" if i<8 else "loop_transition" if i<12 else "success_transition",
                "grip_delta":float((b[:,6]-a[:,6]).mean()),
                "signs":int(((a[:,6]>0)!=(b[:,6]>0)).sum()),
                "xyz_rms":float(np.sqrt(np.mean((b[:,:3]-a[:,:3])**2)))})
for gg in ["empty","holding","loop_transition","success_transition"]:
    subset=[s for s in s47 if s["group"]==gg]
    print(gg,"grip mean delta",round(np.mean([s["grip_delta"] for s in subset]),5),
          "sign changes",sum(s["signs"] for s in subset),
          "XYZ RMS",round(np.mean([s["xyz_rms"] for s in subset]),5))
discovery_ids={json.loads(x)["id"] for x in (W/"roll_1.jsonl").read_text().splitlines() if x.strip()}
all_val_ids=set()
for j,panel in enumerate(panels[1:],start=2):
    ids={p["id"] for p in panel}
    assert not(ids&discovery_ids)
    assert not(ids&all_val_ids)
    assert not(ids&{p["id"] for p in panels[0]})
    all_val_ids|=ids
    print("Panel",j,"episodes",len(panel),"scenes",len(ids),"seed set",sorted({p["seed"] for p in panel}))
protocol["acceptance"]={"gain_threshold":"2*max(episode-paired SE, scene-cluster SE)",
                        "loop_failure_delta":"strictly negative",
                        "baseline_success_losses_max":"0.02 * validation episode count"}
(W/"final_candidate_protocol.json").write_text(json.dumps(protocol,indent=2))
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 46: frozen opposite-dose discovery screen\n"
            "- Beta=-2: 31/64 -> 27/64, -6.2 pp (SE 3.0), 0 rescues / 4 losses; loops 21 -> 23.\n"
            "\n## Pass 47: frozen opposite-dose mechanism probe\n"
            "- "+json.dumps(s47)+"\n"
            "- Independent validation panels checked disjoint from discovery, prior validation, and one another. Acceptance threshold specified before their outcomes: gain >2*max(paired SE, scene-cluster SE), fewer loops, and losses <=2% of validation episodes.\n")
print("\nPASS 48 EPISODES:")
print(json.dumps(panels[1],separators=(",",":")))
```
