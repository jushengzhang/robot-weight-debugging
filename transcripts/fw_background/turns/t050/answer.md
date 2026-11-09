I will quantify the fixed-input effect and prepare the paired diagnostic rollout. A reduction in action differences is only a mechanism check; the edit must also improve task success without degrading the matched light controls.
```python
def read_probe(k):
    return json.loads((W/f"probe_{k}/results.json").read_text())
def compare_probe(k, n_pairs=4):
    rows=read_probe(k)
    E=np.stack([np.load(r["action"]) for r in rows])
    O=np.stack([np.asarray(torch.load(r["input"],map_location="cpu",weights_only=False)["action_chunk"])
                for r in rows])
    rms=lambda x: float(np.sqrt(np.mean(np.square(x))))
    base=np.array([rms(O[2*i]-O[2*i+1]) for i in range(n_pairs)])
    new=np.array([rms(E[2*i]-E[2*i+1]) for i in range(n_pairs)])
    summary={"pass":k,"base_pair_rms":base.tolist(),"edited_pair_rms":new.tolist(),
             "aggregate_pair_ratio":float(np.sqrt(np.mean(new**2)/np.mean(base**2))),
             "bg_change_rms":rms(E[:2*n_pairs]-O[:2*n_pairs]),
             "light_change_rms":rms(E[2*n_pairs:]-O[2*n_pairs:]) if len(E)>2*n_pairs else None}
    print(json.dumps(summary,indent=2))
    return summary
s6=compare_probe(6)
(W/"probe6_summary.json").write_text(json.dumps(s6,indent=2))
diagnostic_eps=json.loads((W/"diagnostic_episodes.json").read_text())
print("Diagnostic rollout request:")
print(json.dumps({"type":"rollout","model":"both","episodes":diagnostic_eps},separators=(",",":")))
```