The half-strength aperture edit gained only 1.6 points with a 3.0-point paired standard error, so it is not yet a demonstrated repair. Its gains were concentrated in closed-gripper starts. I will inspect the changed episodes and saved activations before deciding whether a stronger dose is justified.
```python
def readroll(k):
    return [json.loads(s) for s in (W/f"roll_{k}.jsonl").read_text().splitlines() if s.strip()]
def paired_rows(k):
    groups = defaultdict(dict)
    for r in readroll(k):
        groups[(r["id"],r["grip"],r["seed"])][r["model"]] = r
    return [(key, v["original"],v["edited"]) for key,v in groups.items()
            if "original" in v and "edited" in v]
for key,a,b in paired_rows(17):
    if a["success"] != b["success"]:
        print("WIN" if b["success"] else "LOSS", key[:2], a["level"], a["suite"],
              "flips",a["grip_flips"],b["grip_flips"],"steps",a["steps"],b["steps"],
              a["instruction"][:75])
for k in [8,15]:
    p = W/f"probe_{k}"
    fs = sorted(p.iterdir())
    print("probe",k,"extras",[(f.name,f.stat().st_size) for f in fs if not f.name.endswith((".npy",".npz"))])
    npzs = list(p.glob("*acts.npz"))
    if npzs:
        z=np.load(npzs[0])
        print(" npz",npzs[0].name,[(n,z[n].shape) for n in z.files])
for name in ["roll_1_acts/index.json","roll_4_acts/index.json"]:
    p=W/name
    print(name,p.read_text()[:1600])
with (W/"notes.md").open("a") as f:
    f.write("\n## Pass17: visual aperture direction, alpha0.5\n"
            "Paired development126:72 vs70 successes (+1.6+/-3.0pp SE),8 wins/6 losses; loops44 vs46. "
            "Close starts +6 successes, open -4. Not a demonstrated improvement. "
            "Current projector edit changes W by2.6% Frobenius, with LN-offset-preserving bias correction.\n")
```
