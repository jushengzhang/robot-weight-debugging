I’ll quantify the detector scores on the saved clean, matched-prefix, and novel-tag context vectors. This will tell us whether there is room to lower the threshold while still separating tag-like features from useful task text.
```python
torch.set_num_threads(12)
geo = torch.load(W / "dense_gate_geometry.pt", weights_only=False)
X = torch.from_numpy(np.load(W / "context_dense_56x300x4096.npy"))
dirs = geo["dirs"].float()
def qs(t):
    return [round(float(z), 5) for z in torch.quantile(t.float().reshape(-1), torch.tensor([0., .1, .5, .9, .99, 1.]))]
for label, arr in [("clean+language", X[24:].reshape(-1,4096)),
                   ("matched raw prefix", geo["raw_prefix"]),
                   ("novel raw", geo["novel"])]:
    scores = (arr.float() @ dirs.T).amax(-1)
    print(label, "score quantiles [min,p10,p50,p90,p99,max]", qs(scores),
          "fractions > .10/.15/.20", [(t,round(float((scores>t).float().mean()),4)) for t in [.10,.15,.20]])
print("By category, novel context scores:")
for c in sorted(set(m[1] for m in geo["novel_meta"])):
    idx = [i for i,m in enumerate(geo["novel_meta"]) if m[1]==c]
    print(c, qs((geo["novel"][idx] @ dirs.T).amax(-1)))
print("\nHELPER CODE:")
print((W / "analysis_helpers.py").read_text()[:4800])
print("\nMETA STRUCTURES:")
print("replay_meta", type(rmeta), str(rmeta)[:550])
print("probe_items", type(pitems), str(pitems)[:550])
```
