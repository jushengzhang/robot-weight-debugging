I will compare zero, half, and full edit strength on the fixed inputs, then run the half-strength model on the original 80-case goal diagnostic set. This is a mechanistic ablation, not a new round of tuning; the confirmed full recipe remains the intended final edit.
```python
P32={(r["idx"],r["arm"]):r for r in probe_results(32)}
key="mean::expert.17.resid::action"
center=np.mean([np.load(P30[(i,"orig")]["activations"])[key] for i in [1,3,5,7,9]],axis=0)
dose_summary=[]
for i in range(0,12,2):
    ar=np.load(P30[(i+1,"orig")]["action"])
    target=np.load(P30[(i+1,"orig")]["activations"])[key]-center
    sources=[P30[(i,"orig")],P32[(i,"edit")],P30[(i,"edit")]]
    rec={"case":probe_cases[i]["label"],
         "dose":[0,0.5,1],
         "xyz_rms":[round(float(np.sqrt(np.mean((np.load(r["action"])[:,:3]-ar[:,:3])**2))),4) for r in sources],
         "expert17_centered_cos":[round(cosine(np.load(r["activations"])[key]-center,target),3) for r in sources]}
    dose_summary.append(rec)
    print(rec)
json.dump(dose_summary,(W/"F_dose_probe_summary.json").open("w"),indent=2)
dose_eps=[{k:r[k] for k in ["suite","id","seed"]} for r in read_roll(1)]
assert len(dose_eps)==80
json.dump(dose_eps,(W/"dose_goal_episodes.json").open("w"))
print("DOSE EPISODES",json.dumps(dose_eps,separators=(",",":")))
with (W/"notes.md").open("a") as f:
    f.write("\n## Attention mechanism (pass31)\nRelevant text is attended before repair; layer6 non-BOS text mass "
            "original->F: cooker .141->.124, basin .196->.138, grape .228->.211, spread .254->.228. "
            "Thus not an attention-amplification repair. Common language-reading heads include L6 H6/H2. "
            "No attention head weights edited; causal localization is sparse lexical embeddings.\n"
            "Pass32 half-dose fixed-input results in F_dose_probe_summary.json. "
            "Half-dose still active for upcoming80-case diagnostic pass33; restore F immediately afterward.\n")
```
