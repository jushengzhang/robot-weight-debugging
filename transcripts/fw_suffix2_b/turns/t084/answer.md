I’ll check the projection’s spectrum and clean-context leakage before choosing a stronger dose. This is a static analysis of saved activations and weights—not a network run or weight fitting—and should show whether adding directions would mainly amplify poorly supported noise.
```python
exec((W / "analysis_helpers.py").read_text(), globals())
X = torch.from_numpy(np.load(W / "context_features_72x30x4096.npy")).double()
P = art["rich_context_projection.pt"]
U = P["U"].double()
B = P["B_nearest"].double()
D = P["D"].double()
s = P["S"].double()
cs = art["clean_context_svd.pt"]["Csingular"].double()
print("Contrast singular values at selected indices:")
for r in (32, 64, 128, 192, 256, 320, 384, 448, 512, 600, 700):
    print(r, round(float(s[r-1]), 5),
          "energy", round(float(s[:r].square().sum()/s.square().sum()), 6),
          "B_RMS_gain", round(float(B[:, :r].square().sum()/r)**.5, 3))
print("\nClean singular spectrum:")
for r in (256, 384, 448, 512, 576, 640, 700, 720, 800):
    print(r, float(cs[r-1]), float(cs[:r].square().sum()/cs.square().sum()))
print("\nProjection relative change on measured context samples:")
for r in (128, 192, 256, 320):
    out = {}
    for label, z in (("raw", X[:24]), ("clean", X[24:48]), ("language", X[48:])):
        z = z.flatten(0, 1)
        delta = (z @ U[:, :r]) @ B[:, :r].T
        out[label] = round(float(delta.norm()/z.norm()), 6)
    print(r, out)

print("\nDEVELOPMENT ROLLOUTS")
for k in (22, 23, 26, 28, 29):
    print(k, {a:b for a,b in paired_rolls([k]).items() if a != "by_category"})
print("\nSPLIT KEYS", [(k,len(v)) for k,v in sp.items()])
print("DIAGNOSTIC ENTRY", sp["diagnostic"][0])
with (W / "notes.md").open("a") as f:
    f.write("\nPass28 broad hybrid language development96: original65/96, edited65/96; "
            +json.dumps({k:v for k,v in paired_rolls([28]).items() if k!="by_category"})+"\n")
    f.write("\nPass29 broad hybrid clean development96: original66/96, edited64/96; "
            "-2.08pp SE3.91, 6gains/8losses. Candidate is not yet validated. "
            "All reserved validation sets remain untouched at pass29.\n")
```
