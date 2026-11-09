I will measure whether the intervention produced its predicted attention change, and compare mild/severe action discrepancies before and after it. I will then launch a paired rollout on the already-fixed discovery camera panel; attention changes alone will not be treated as a success.```python
log_pass(4,"probe","Causal attention check of selective L29 Q/K gain cap","l29_qk_cap4",
         {"inputs":4,"noise_seed":46001})
p4=json.loads((W/"probe_4"/"results.json").read_text())
m4=np.stack([np.load(r["attention"]["29"]).astype(np.float64) for r in p4]).mean(2)
tot4=m4.sum(2)
agent4=m4[:,:,agent_idx].sum(2)/(tot4+1e-12)
pcond4=m4/(tot4[:,:,None]+1e-12)
ent4=-(pcond4*np.log(pcond4+1e-12)).sum(2)/math.log(98)
print("L29 candidate attention changes, original -> cap4, averaged over 4 inputs:")
for h in (4,7,8,17):
    print(h,"agent fraction",round(float(head_map_stats[29]["agent_fraction"][:,h].mean()),4),
          "->",round(float(agent4[:,h].mean()),4),
          "entropy",round(float(head_map_stats[29]["entropy"][:,h].mean()),4),
          "->",round(float(ent4[:,h].mean()),4))
act3=np.stack([np.load(r["action"]) for r in p3])
act4=np.stack([np.load(r["action"]) for r in p4])
for p in (0,2):
    orig_rms=np.sqrt(np.mean((act3[p+1,:10,:3]-act3[p,:10,:3])**2))
    edit_rms=np.sqrt(np.mean((act4[p+1,:10,:3]-act4[p,:10,:3])**2))
    print("Pair",p//2,"XYZ view discrepancy",round(float(orig_rms),5),"->",round(float(edit_rms),5))
print("Overall first10 XYZ edit RMS:",round(float(np.sqrt(np.mean((act4[:,:10,:3]-act3[:,:10,:3])**2))),5))
print("Screen episodes:",json.dumps(screen_cam_episodes,separators=(",",":")))
```
