I will summarize the attention and action changes from the rank-one probe. If it has the intended selective effect, the next step is the fixed paired rollout screen—not a claim of recovery improvement from attention alone.
```python
pa36=probe_actions(36)
probe36_summary=[]
for i in range(4):
    masses=[]
    for arm in ["orig","edit"]:
        m=np.load(W/"probe_36"/f"probe_{arm}_{i}_attn_L16.npy")[3].mean(0)
        masses.append([m[:256].sum(),m[256:512].sum(),m[768:].sum(),1-m.sum(),
                       m[775] if i>=2 else m[769]])
    a,b=pa36[i]["orig"],pa36[i]["edit"]
    rms=float(np.sqrt(np.mean((b[:,:3]-a[:,:3])**2)))
    print(i,"B/W/lang/self/verb",np.round(masses,4).tolist(),"XYZ RMS",round(rms,4),
          "first5 delta",np.round((b[:5]-a[:5]).mean(0),4).tolist())
    probe36_summary.append({"input":i,"masses":np.array(masses).tolist(),"xyz_rms":rms})
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass 36: dominant query-mode causal probe\n"
            "- E16H3 leading singular mode carries 5.7% of head weight energy but 36-49% of observed mean-query energy.\n"
            "- Candidate halves only this rank-one mode; other query modes unchanged.\n"
            "- Probe summary: "+json.dumps(probe36_summary)+"\n")
```
