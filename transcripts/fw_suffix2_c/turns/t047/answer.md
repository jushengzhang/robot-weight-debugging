The probe API requires individual registered input files rather than a directory. I’ll recover the exact file list and rebuild the comparison helper before resubmitting; the refused request used no pass.
```python
paths = json.loads((W/"replay_paths.json").read_text())
meta = json.loads((W/"replays_meta.json").read_text())
print("PATH RANGES", [(i,paths[i]) for i in [0,119,120,239,240,271]])
print("PROBE META FILES", [p.name for p in (W/"probe_15").iterdir() if p.suffix not in (".npy", ".npz")])
for f in (W/"probe_15").glob("*.json"):
    print(f.name, f.read_text()[:800])
def read_actions(k, normalized=False):
    arm = "orig" if k==5 else "edit"
    suffix = "action_norm.npy" if normalized else "action.npy"
    fs = [W/f"probe_{k}"/f"probe_{arm}_{i}_{suffix}" for i in range(272)]
    return np.stack([np.load(f) for f in fs])
good = np.array([i for i in range(120) if i not in [22,38,59,115]])
for norm in [False,True]:
    z=read_actions(5,norm)
    print("normalized",norm,"shape",z.shape,"tag gap",np.sqrt(np.mean((z[good]-z[120+good])**2)))
# Save reusable comparison helper, with no model execution.
helper = '''
import json, numpy as np
from pathlib import Path
W = Path(WORK)
meta = json.loads((W/"replays_meta.json").read_text())
good = np.array([i for i in range(120) if i not in [22,38,59,115]])
def read_actions(k, normalized=False):
    arm = "orig" if k==5 else "edit"
    suffix = "action_norm.npy" if normalized else "action.npy"
    return np.stack([np.load(W/f"probe_{k}"/f"probe_{arm}_{i}_{suffix}") for i in range(272)])
def rms(a): return float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2)))
def compare(k, normalized=False, log=True):
    b,e=read_actions(5,normalized),read_actions(k,normalized)
    delta=b[good]-b[120+good]
    gap=rms(delta)
    out={"pass":k, "baseline_gap":gap,
         "raw_to_clean":rms(e[good]-b[120+good]),
         "gap_ratio":rms(e[good]-b[120+good])/gap,
         "edited_raw_clean_gap":rms(e[good]-e[120+good]),
         "clean_drift":rms(e[120+good]-b[120+good]),
         "language_drift":rms(e[240:]-b[240:]),
         "raw_change":rms(e[good]-b[good]),
         "correction_fraction":float(-np.sum((e[good]-b[good])*delta)/np.sum(delta**2))}
    print(json.dumps(out))
    for cat in sorted(set(m["category"] for m in meta[:120])):
        g=np.array([i for i in good if meta[i]["category"]==cat])
        print(cat,"gap_ratio",round(rms(e[g]-b[120+g])/rms(b[g]-b[120+g]),4),
              "clean_drift",round(rms(e[120+g]-b[120+g]),4))
    for label,t,d in [("all7_first10",slice(0,10),slice(None)),
                      ("arm6_first10",slice(0,10),slice(0,6)),
                      ("arm6_full",slice(None),slice(0,6)),
                      ("gripper_full",slice(None),slice(6,7))]:
        orig=rms(b[good,t,d]-b[120+good,t,d])
        print(label,"gap_ratio",round(rms(e[good,t,d]-b[120+good,t,d])/orig,4),
              "clean",round(rms(e[120+good,t,d]-b[120+good,t,d]),5))
    if log:
        with (W/"notes.md").open("a") as f: f.write("\\nProbe metrics: "+json.dumps(out)+"\\n")
    return out
'''
(W/"analysis_helpers.py").write_text(helper)
exec(helper)
compare(15, log=False)
```
