The recent candidate probes used a 128-input screen: 48 matched tagged/clean pairs plus 32 paraphrases. I’ll preserve that ordering for the next test. The larger replay set remains available for later checks.
```python
screen_ids = json.loads((W/"screen_ids.json").read_text())
screen_paths = [paths[i] for i in screen_ids] + [paths[120+i] for i in screen_ids] + paths[240:]
ref = np.load(W/"screen_ref.npy")
print("REF",ref.shape,"gap",float(np.sqrt(np.mean((ref[:48]-ref[48:96])**2))))
print("ORIGINAL_MATCH",
      float(np.max(np.abs(ref-read_actions(5,False)[screen_ids+[120+i for i in screen_ids]+list(range(240,272))]))))
helper2 = '''
import json, numpy as np
from pathlib import Path
W=Path(WORK)
screen_ids=json.loads((W/"screen_ids.json").read_text())
meta=json.loads((W/"replays_meta.json").read_text())
ref=np.load(W/"screen_ref.npy")
def rms(a): return float(np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2)))
def screen_actions(k):
    return np.stack([np.load(W/f"probe_{k}"/f"probe_edit_{i}_action.npy") for i in range(128)])
def compare_screen(k,log=True):
    b,e=ref,screen_actions(k)
    delta=b[:48]-b[48:96]
    gap=rms(delta)
    out={"pass":k,"baseline_gap":gap,"raw_to_clean":rms(e[:48]-b[48:96]),
         "gap_ratio":rms(e[:48]-b[48:96])/gap,
         "edited_raw_clean_gap":rms(e[:48]-e[48:96]),
         "clean_drift":rms(e[48:96]-b[48:96]),
         "language_drift":rms(e[96:]-b[96:]),
         "raw_change":rms(e[:48]-b[:48]),
         "correction_fraction":float(-np.sum((e[:48]-b[:48])*delta)/np.sum(delta**2))}
    print(json.dumps(out))
    for cat in sorted(set(meta[i]["category"] for i in screen_ids)):
        g=np.array([j for j,i in enumerate(screen_ids) if meta[i]["category"]==cat])
        print(cat,"gap_ratio",round(rms(e[g]-b[48+g])/rms(b[g]-b[48+g]),4),
              "clean_drift",round(rms(e[48+g]-b[48+g]),4))
    for label,t,d in [("all7_first10",slice(0,10),slice(None)),
                      ("arm6_first10",slice(0,10),slice(0,6)),
                      ("arm6_full",slice(None),slice(0,6)),
                      ("gripper_full",slice(None),slice(6,7))]:
        orig=rms(b[:48,t,d]-b[48:96,t,d])
        print(label,"gap_ratio",round(rms(e[:48,t,d]-b[48:96,t,d])/orig,4),
              "clean",round(rms(e[48:96,t,d]-b[48:96,t,d]),5))
    if log:
        with (W/"notes.md").open("a") as f: f.write("\\nProbe metrics: "+json.dumps(out)+"\\n")
    return out
'''
(W/"screen_helpers.py").write_text(helper2)
exec(helper2)
compare_screen(15,log=False)
print("SCREEN INPUTS", json.dumps(screen_paths))
```
